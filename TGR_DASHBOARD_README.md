# TGR Data Enrichment Dashboard

A comprehensive web-based dashboard for monitoring and controlling the TGR (The Greyhound Racers) data enrichment system.

## Features

### 🎯 System Overview
- **System Health Monitoring**: Real-time status, uptime, and success rates
- **Data Quality Metrics**: Completeness, freshness, and overall quality scores
- **Job Processing Stats**: 24-hour job counts, queue size, active workers
- **Cache Performance**: Hit rates, total entries, active cache statistics

### 📊 Interactive Charts
- **Performance Trends**: Line chart showing jobs processed, success rates, and cache performance over time
- **Job Status Distribution**: Doughnut chart displaying job status breakdown (completed, pending, running, failed)
- **Time Range Selection**: Switch between 1 hour, 24 hours, and 7-day views

### 🔧 Service Management
- **Monitoring Dashboard**: Real-time health monitoring service
- **Enrichment Service**: Multi-threaded data processing engine
- **Intelligent Scheduler**: Automated job scheduling system
- **Database System**: Enhanced schema with TGR performance tables

### 🚨 Alerts & Activity
- **System Alerts**: Real-time notifications for system events
- **Activity Timeline**: Recent system activities with filtering options
- **Real-time Updates**: WebSocket-powered live updates

### ⚡ Real-time Features
- WebSocket connection for live data updates
- Automatic refresh every 30 seconds
- Connection status indicator
- Live job creation notifications

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Dashboard

```bash
python start_dashboard.py
```

### 3. Access the Dashboard

Open your browser and navigate to:
- **Local Access**: http://localhost:5000
- **Network Access**: http://[your-ip]:5000

## Architecture

### Backend Components
- **Flask Server** (`tgr_dashboard_server.py`): Main web server with REST API endpoints
- **WebSocket Support**: Real-time bidirectional communication
- **TGR Integration**: Direct integration with existing TGR modules
- **Database Connectivity**: SQLite database integration

### Frontend Components
- **HTML Dashboard** (`frontend/index.html`): Responsive web interface
- **CSS Styling** (`frontend/css/dashboard.css`): Modern, professional styling
- **JavaScript Controllers**:
  - `dashboard.js`: Main controller and UI management
  - `charts.js`: Chart rendering with Chart.js
  - `api.js`: API communication and WebSocket handling

### API Endpoints

#### System Status
- `GET /api/v1/status/system` - Get overall system status
- `GET /health` - Health check endpoint

#### Service Management  
- `GET /api/v1/services/status` - Get all service statuses
- `POST /api/v1/services/start` - Start all services
- `POST /api/v1/services/stop` - Stop all services

#### Monitoring & Alerts
- `GET /api/v1/alerts` - Get system alerts
- `DELETE /api/v1/alerts` - Clear all alerts
- `GET /api/v1/activity` - Get recent system activity

#### Analytics
- `GET /api/v1/metrics/performance?range=24h` - Get performance metrics for charts

#### Job Management
- `POST /api/v1/jobs` - Create new enrichment job

## Configuration

### Environment Variables
- `FLASK_ENV`: Set to 'development' for development mode
- `FLASK_DEBUG`: Set to '0' for production

### Database Configuration
The dashboard expects the SQLite database `greyhound_racing_data.db` in the project root.

### TGR Module Integration
The dashboard automatically detects and integrates with existing TGR modules:
- `tgr_monitoring_dashboard.py`
- `tgr_enrichment_service.py` 
- `tgr_service_scheduler.py`

## Features in Detail

### Real-time Data Updates
- WebSocket connection provides live updates
- Automatic fallback to mock data if TGR modules unavailable
- Connection status indicator shows real-time connectivity

### Responsive Design
- Works on desktop, tablet, and mobile devices
- Modern CSS Grid and Flexbox layouts
- Professional color scheme with CSS custom properties

### Service Control
- Start/stop individual or all services
- Real-time service status updates
- Service health indicators with color-coded status

### Interactive Charts
- Performance trends with multiple metrics
- Job status distribution with dynamic data
- Responsive charts that adapt to container size
- Export functionality for chart images

### Alert System
- Color-coded alert levels (info, warning, danger, success)
- Real-time alert notifications
- Bulk alert clearing functionality

### Activity Monitoring
- Filterable activity timeline
- Real-time activity feed
- Activity type categorization (jobs, health, alerts, system)

## Development

### Project Structure
```
tgr-dashboard/
├── tgr_dashboard_server.py     # Flask backend server
├── start_dashboard.py          # Startup script
├── requirements.txt            # Python dependencies
├── frontend/
│   ├── index.html             # Main dashboard page
│   ├── css/
│   │   └── dashboard.css      # Dashboard styling
│   └── js/
│       ├── dashboard.js       # Main controller
│       ├── charts.js         # Chart management
│       └── api.js           # API communication
└── TGR_DASHBOARD_README.md    # This file
```

### Adding New Features

1. **Backend**: Add new API endpoints in `tgr_dashboard_server.py`
2. **Frontend**: Update the corresponding JavaScript modules
3. **UI**: Modify `index.html` and `dashboard.css` for new interface elements

### Mock Data
The dashboard includes comprehensive mock data fallbacks for development when TGR modules are not available.

## Troubleshooting

### Common Issues

1. **Dependencies not installed**
   ```bash
   pip install flask flask-cors flask-socketio python-socketio python-engineio
   ```

2. **Port already in use**
   - Change port in `tgr_dashboard_server.py` (default: 5000)
   - Kill existing processes: `lsof -ti:5000 | xargs kill`

3. **Database not found**
   - Ensure `greyhound_racing_data.db` exists in project root
   - Dashboard will show mock data if database unavailable

4. **TGR modules not found**
   - Dashboard falls back to mock data automatically
   - Ensure TGR module files are in the project root

### Logs
- Server logs are printed to console
- Browser console shows client-side debugging info
- Check network tab for API communication issues

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Performance

- Optimized for real-time updates with WebSockets
- Efficient chart rendering with Chart.js
- Responsive design with CSS Grid/Flexbox
- Minimal JavaScript footprint

## Security

- CORS enabled for development
- No authentication implemented (add as needed)
- Local network access available
- WebSocket security considerations apply

## Future Enhancements

- [ ] User authentication and authorization
- [ ] Historical data analysis
- [ ] Custom dashboard layouts
- [ ] Export functionality for reports
- [ ] Mobile app companion
- [ ] Advanced alerting rules
- [ ] Integration with external monitoring systems

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review browser console for client errors
3. Check server logs for backend issues
4. Ensure all dependencies are properly installed

## License

Part of the TGR (The Greyhound Racers) data enrichment system.
