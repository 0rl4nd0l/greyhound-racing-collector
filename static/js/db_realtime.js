/**
 * Database Realtime Monitoring
 * Connects to WebSockets or Server-Sent Events for real-time database metrics
 */

class DatabaseRealtime {
  constructor(containerElement) {
    this.container = containerElement;
    if (!this.container) return;

    this.connection = null;
    this.reconnectTimer = null;
    this.reconnectDelay = 1000; // Start with 1s, will increase with backoff
    this.maxReconnectDelay = 30000; // Max 30s between reconnect attempts
    this.eventBuffer = []; // Store recent events for replay on reconnect
    this.bufferSize = 50; // Max events to keep
    this.paused = false;
    this.lastConnectedAt = null;
    
    this.healthScore = 100;
    this.queryRate = 0;
    this.avgLatency = 0;
    this.connectionCount = 0;
    this.errorRate = 0;
    
    this.init();
  }

  init() {
    this.renderInitialState();
    try {
      if (window && window.E2E_DISABLE_REALTIME) {
        // Skip establishing realtime connections in E2E/TESTING mode
        return;
      }
    } catch (e) {}
    this.connect();
    this.setupEventListeners();
  }

  renderInitialState() {
    this.container.innerHTML = `
      <div class="realtime-header">
        <div class="realtime-status" id="connection-status">
          <span class="db-status-badge info">
            <span class="db-status-dot"></span>
            Connecting...
          </span>
        </div>
        <div class="realtime-controls">
          <button class="db-button secondary" id="toggle-pause">
            <i class="fas fa-pause"></i> Pause
          </button>
          <button class="db-button secondary" id="clear-events">
            <i class="fas fa-eraser"></i> Clear
          </button>
        </div>
      </div>
      
      <div class="realtime-metrics">
        <div class="status-overview">
          <div class="status-card" id="health-card">
            <div class="status-card-header">
              <div class="status-card-icon"><i class="fas fa-heartbeat"></i></div>
              <h4 class="status-card-title">Health</h4>
            </div>
            <div class="status-card-value">100</div>
            <div class="status-card-change healthy">Healthy</div>
          </div>
          
          <div class="status-card" id="qps-card">
            <div class="status-card-header">
              <div class="status-card-icon"><i class="fas fa-tachometer-alt"></i></div>
              <h4 class="status-card-title">Queries/sec</h4>
            </div>
            <div class="status-card-value">0</div>
            <div class="status-card-change">
              <canvas class="mini-sparkline" width="50" height="20"></canvas>
            </div>
          </div>
          
          <div class="status-card" id="latency-card">
            <div class="status-card-header">
              <div class="status-card-icon"><i class="fas fa-clock"></i></div>
              <h4 class="status-card-title">Avg Latency</h4>
            </div>
            <div class="status-card-value">0 ms</div>
            <div class="status-card-change">
              <canvas class="mini-sparkline" width="50" height="20"></canvas>
            </div>
          </div>
          
          <div class="status-card" id="conn-card">
            <div class="status-card-header">
              <div class="status-card-icon"><i class="fas fa-plug"></i></div>
              <h4 class="status-card-title">Connections</h4>
            </div>
            <div class="status-card-value">0</div>
            <div class="status-card-change">
              <span class="text-muted">Pool: 0/0</span>
            </div>
          </div>
        </div>
      </div>
      
      <div class="realtime-events">
        <h4>Recent Events <span class="count-badge" id="event-count">0</span></h4>
        <div class="event-list" id="event-list" role="log" aria-live="polite">
          <div class="event-empty">No events yet</div>
        </div>
      </div>
    `;
  }

  setupEventListeners() {
    const pauseButton = this.container.querySelector('#toggle-pause');
    const clearButton = this.container.querySelector('#clear-events');
    
    if (pauseButton) {
      pauseButton.addEventListener('click', () => {
        this.togglePause();
      });
    }
    
    if (clearButton) {
      clearButton.addEventListener('click', () => {
        this.clearEvents();
      });
    }
    
    // Handle visibility change to reconnect when tab becomes visible
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        if (!this.connection || this.connection.readyState > 1) {
          this.connect();
        }
      }
    });
  }

  togglePause() {
    this.paused = !this.paused;
    
    const pauseButton = this.container.querySelector('#toggle-pause');
    if (pauseButton) {
      pauseButton.innerHTML = this.paused 
        ? '<i class="fas fa-play"></i> Resume'
        : '<i class="fas fa-pause"></i> Pause';
    }
    
    this.updateConnectionStatus();
  }

  clearEvents() {
    const eventList = this.container.querySelector('#event-list');
    if (eventList) {
      eventList.innerHTML = '<div class="event-empty">No events yet</div>';
      this.updateEventCount(0);
    }
  }

  updateConnectionStatus(status = null) {
    const statusElement = this.container.querySelector('#connection-status');
    if (!statusElement) return;
    
    let statusHtml = '';
    
    if (this.paused) {
      statusHtml = `
        <span class="db-status-badge warning">
          <span class="db-status-dot"></span>
          Paused
        </span>
      `;
    } else if (status === 'connected') {
      this.lastConnectedAt = new Date();
      statusHtml = `
        <span class="db-status-badge healthy">
          <span class="db-status-dot"></span>
          Connected
        </span>
      `;
    } else if (status === 'connecting') {
      statusHtml = `
        <span class="db-status-badge info">
          <span class="db-status-dot"></span>
          Connecting...
        </span>
      `;
    } else if (status === 'disconnected') {
      statusHtml = `
        <span class="db-status-badge error">
          <span class="db-status-dot"></span>
          Disconnected
        </span>
      `;
    } else if (status === 'error') {
      statusHtml = `
        <span class="db-status-badge error">
          <span class="db-status-dot"></span>
          Connection Error
        </span>
      `;
    }
    
    if (statusHtml) {
      statusElement.innerHTML = statusHtml;
    }
  }

  connect() {
    // Clear any pending reconnect
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    // Update status to connecting
    this.updateConnectionStatus('connecting');

    // Try WebSocket first
    if (this.supportsWebSockets()) {
      this.connectWebSocket();
    } else {
      // Fall back to SSE
      this.connectSSE();
    }
  }

  supportsWebSockets() {
    return 'WebSocket' in window && window.WebSocket !== null;
  }

  connectWebSocket() {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/db`;
      
      this.connection = new WebSocket(wsUrl);
      
      this.connection.onopen = () => {
        this.onConnectionOpen();
      };
      
      this.connection.onmessage = (event) => {
        this.onMessage(event.data);
      };
      
      this.connection.onclose = () => {
        this.onConnectionClosed();
      };
      
      this.connection.onerror = (error) => {
        this.onConnectionError(error);
      };
    } catch (error) {
      console.error('WebSocket connection error:', error);
      this.connectSSE(); // Fall back to SSE
    }
  }

  connectSSE() {
    try {
      const sseUrl = `/api/db/events`;
      
      if (typeof EventSource === 'undefined') {
        throw new Error('EventSource not supported');
      }
      
      this.connection = new EventSource(sseUrl);
      
      this.connection.onopen = () => {
        this.onConnectionOpen();
      };
      
      this.connection.onmessage = (event) => {
        this.onMessage(event.data);
      };
      
      this.connection.onerror = (error) => {
        this.onConnectionError(error);
        this.connection.close();
      };
    } catch (error) {
      console.error('SSE connection error:', error);
      this.scheduleReconnect();
    }
  }

  onConnectionOpen() {
    this.updateConnectionStatus('connected');
    this.reconnectDelay = 1000; // Reset backoff delay
  }

  onConnectionClosed() {
    this.updateConnectionStatus('disconnected');
    this.scheduleReconnect();
  }

  onConnectionError(error) {
    console.error('Realtime connection error:', error);
    this.updateConnectionStatus('error');
    this.scheduleReconnect();
  }

  onMessage(data) {
    if (this.paused) return;
    
    try {
      const event = typeof data === 'string' ? JSON.parse(data) : data;
      
      // Add to buffer and trim if needed
      this.eventBuffer.push(event);
      if (this.eventBuffer.length > this.bufferSize) {
        this.eventBuffer.shift();
      }
      
      // Process the event
      this.processEvent(event);
    } catch (error) {
      console.error('Error processing event:', error, data);
    }
  }

  processEvent(event) {
    // Process based on event type
    switch (event.type) {
      case 'metric':
        this.updateMetrics(event.payload);
        break;
      case 'slow_query':
        this.addSlowQueryEvent(event);
        break;
      case 'connection':
        this.updateConnectionMetric(event.payload);
        break;
      case 'error':
        this.addErrorEvent(event);
        break;
      case 'heartbeat':
        // Just update connection status, no UI changes
        this.updateConnectionStatus('connected');
        break;
    }
  }

  updateMetrics(metrics) {
    // Update health score
    if (metrics.health_score !== undefined) {
      this.healthScore = metrics.health_score;
      this.updateHealthCard(metrics.health_score);
    }
    
    // Update query rate
    if (metrics.queries_per_second !== undefined) {
      this.queryRate = metrics.queries_per_second;
      this.updateQueryRateCard(metrics.queries_per_second);
    }
    
    // Update latency
    if (metrics.avg_latency !== undefined) {
      this.avgLatency = metrics.avg_latency;
      this.updateLatencyCard(metrics.avg_latency);
    }
  }

  updateHealthCard(score) {
    const card = this.container.querySelector('#health-card');
    if (!card) return;
    
    const valueElement = card.querySelector('.status-card-value');
    const statusElement = card.querySelector('.status-card-change');
    
    if (valueElement) {
      valueElement.textContent = score;
    }
    
    if (statusElement) {
      // Remove all status classes
      statusElement.classList.remove('healthy', 'warning', 'error');
      
      // Add appropriate class based on score
      if (score >= 80) {
        statusElement.classList.add('healthy');
        statusElement.textContent = 'Healthy';
      } else if (score >= 60) {
        statusElement.classList.add('warning');
        statusElement.textContent = 'Warning';
      } else {
        statusElement.classList.add('error');
        statusElement.textContent = 'Critical';
      }
    }
  }

  updateQueryRateCard(qps) {
    const card = this.container.querySelector('#qps-card');
    if (!card) return;
    
    const valueElement = card.querySelector('.status-card-value');
    const sparkline = card.querySelector('.mini-sparkline');
    
    if (valueElement) {
      valueElement.textContent = qps.toFixed(1);
    }
    
    if (sparkline) {
      this.updateSparkline(sparkline, qps);
    }
  }

  updateLatencyCard(latency) {
    const card = this.container.querySelector('#latency-card');
    if (!card) return;
    
    const valueElement = card.querySelector('.status-card-value');
    const sparkline = card.querySelector('.mini-sparkline');
    
    if (valueElement) {
      valueElement.textContent = `${latency.toFixed(1)} ms`;
    }
    
    if (sparkline) {
      this.updateSparkline(sparkline, latency);
    }
  }

  updateConnectionMetric(connectionData) {
    const card = this.container.querySelector('#conn-card');
    if (!card) return;
    
    this.connectionCount = connectionData.active || 0;
    const poolSize = connectionData.pool_size || 0;
    const poolUsed = connectionData.pool_used || 0;
    
    const valueElement = card.querySelector('.status-card-value');
    const detailElement = card.querySelector('.status-card-change');
    
    if (valueElement) {
      valueElement.textContent = this.connectionCount;
    }
    
    if (detailElement) {
      detailElement.innerHTML = `<span class="text-muted">Pool: ${poolUsed}/${poolSize}</span>`;
    }
  }

  addSlowQueryEvent(event) {
    this.addEventToList({
      type: 'slow_query',
      icon: 'fas fa-hourglass-half',
      title: 'Slow Query Detected',
      message: `${event.payload.duration}ms: ${this.truncateQuery(event.payload.query)}`,
      timestamp: event.timestamp
    });
  }

  addErrorEvent(event) {
    this.addEventToList({
      type: 'error',
      icon: 'fas fa-exclamation-triangle',
      title: 'Database Error',
      message: event.payload.message || 'Unknown error',
      timestamp: event.timestamp
    });
  }

  addEventToList(eventData) {
    const eventList = this.container.querySelector('#event-list');
    if (!eventList) return;
    
    // Remove empty message if present
    const emptyMessage = eventList.querySelector('.event-empty');
    if (emptyMessage) {
      emptyMessage.remove();
    }
    
    // Create event element
    const eventElement = document.createElement('div');
    eventElement.className = `event-item ${eventData.type}`;
    
    const formattedTime = this.formatTimestamp(eventData.timestamp);
    
    eventElement.innerHTML = `
      <div class="event-header">
        <span class="event-icon"><i class="${eventData.icon}"></i></span>
        <span class="event-title">${eventData.title}</span>
        <span class="event-time">${formattedTime}</span>
      </div>
      <div class="event-body">
        ${eventData.message}
      </div>
    `;
    
    // Add to top of list
    eventList.prepend(eventElement);
    
    // Limit number of visible events
    const events = eventList.querySelectorAll('.event-item');
    const visibleCount = 100; // Max visible events
    
    if (events.length > visibleCount) {
      for (let i = visibleCount; i < events.length; i++) {
        events[i].remove();
      }
    }
    
    // Update event count
    this.updateEventCount(events.length);
  }

  updateEventCount(count) {
    const countElement = this.container.querySelector('#event-count');
    if (countElement) {
      countElement.textContent = count;
    }
  }

  updateSparkline(canvas, newValue) {
    if (!canvas || !canvas.getContext) return;
    
    // Initialize data if needed
    if (!canvas.sparklineData) {
      canvas.sparklineData = new Array(20).fill(0);
    }
    
    // Add new value and shift
    canvas.sparklineData.push(newValue);
    canvas.sparklineData.shift();
    
    // Find min/max for scaling
    const max = Math.max(...canvas.sparklineData, 1); // Ensure at least 1 to avoid division by zero
    
    // Get canvas context
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw sparkline
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.8)'; // Primary blue
    ctx.lineWidth = 1.5;
    
    const width = canvas.width;
    const height = canvas.height;
    const step = width / (canvas.sparklineData.length - 1);
    
    // Start at first point
    ctx.moveTo(0, height - (canvas.sparklineData[0] / max) * height);
    
    // Draw lines to each point
    for (let i = 1; i < canvas.sparklineData.length; i++) {
      const x = i * step;
      const y = height - (canvas.sparklineData[i] / max) * height;
      ctx.lineTo(x, y);
    }
    
    ctx.stroke();
    
    // Draw a dot at the last point
    const lastX = (canvas.sparklineData.length - 1) * step;
    const lastY = height - (canvas.sparklineData[canvas.sparklineData.length - 1] / max) * height;
    
    ctx.beginPath();
    ctx.arc(lastX, lastY, 2, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(59, 130, 246, 1)';
    ctx.fill();
  }

  scheduleReconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    
    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, this.reconnectDelay);
    
    // Increase delay with backoff (max 30s)
    this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, this.maxReconnectDelay);
  }

  formatTimestamp(timestamp) {
    if (!timestamp) return '';
    
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString();
    } catch (e) {
      return timestamp;
    }
  }

  truncateQuery(query, maxLength = 60) {
    if (!query) return 'Unknown query';
    if (query.length <= maxLength) return query;
    return query.substring(0, maxLength) + '...';
  }

  disconnect() {
    if (this.connection) {
      if (this.connection.close) {
        this.connection.close();
      } else if (this.connection.abort) {
        this.connection.abort();
      }
    }
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
}

// Make available globally
window.DatabaseRealtime = DatabaseRealtime;

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = DatabaseRealtime;
}
