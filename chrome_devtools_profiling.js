/**
 * Chrome DevTools Performance Profiling Script
 * ============================================
 * 
 * This script demonstrates how to use Chrome DevTools for frontend
 * performance monitoring and bottleneck detection.
 * 
 * Usage:
 * 1. Open Chrome DevTools (F12)
 * 2. Go to Console tab
 * 3. Paste and run this script
 * 4. Navigate through the application
 * 5. Check Performance tab for results
 */

(function() {
    'use strict';
    
    console.log('🚀 Starting Chrome DevTools Performance Monitoring');
    
    // Performance monitoring configuration
    const config = {
        measureApiCalls: true,
        measurePageLoad: true,
        measureUserInteractions: true,
        logThreshold: 100, // Log operations slower than 100ms
    };
    
    // Performance metrics storage
    const performanceMetrics = {
        apiCalls: [],
        pageLoads: [],
        userInteractions: [],
        resourceLoading: []
    };
    
    /**
     * Measure API call performance
     */
    function measureApiPerformance() {
        if (!config.measureApiCalls) return;
        
        // Override fetch to measure API calls
        const originalFetch = window.fetch;
        window.fetch = async function(...args) {
            const url = args[0];
            const startTime = performance.now();
            
            try {
                const response = await originalFetch.apply(this, args);
                const endTime = performance.now();
                const duration = endTime - startTime;
                
                const apiMetric = {
                    url,
                    method: args[1]?.method || 'GET',
                    status: response.status,
                    duration: duration.toFixed(2),
                    timestamp: new Date().toISOString(),
                    size: response.headers.get('content-length') || 'unknown'
                };
                
                performanceMetrics.apiCalls.push(apiMetric);
                
                if (duration > config.logThreshold) {
                    console.warn(`🐌 Slow API call detected:`, apiMetric);
                } else {
                    console.log(`⚡ API call completed:`, apiMetric);
                }
                
                return response;
            } catch (error) {
                const endTime = performance.now();
                const duration = endTime - startTime;
                
                console.error(`❌ API call failed:`, {
                    url,
                    duration: duration.toFixed(2),
                    error: error.message
                });
                
                throw error;
            }
        };
    }
    
    /**
     * Measure page load performance
     */
    function measurePageLoadPerformance() {
        if (!config.measurePageLoad) return;
        
        window.addEventListener('load', function() {
            const perfData = performance.getEntriesByType('navigation')[0];
            
            const pageLoadMetric = {
                domContentLoaded: perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart,
                loadEvent: perfData.loadEventEnd - perfData.loadEventStart,
                domComplete: perfData.domComplete - perfData.navigationStart,
                totalLoadTime: perfData.loadEventEnd - perfData.navigationStart,
                dnsLookup: perfData.domainLookupEnd - perfData.domainLookupStart,
                tcpConnect: perfData.connectEnd - perfData.connectStart,
                serverResponse: perfData.responseEnd - perfData.requestStart,
                domProcessing: perfData.domComplete - perfData.responseEnd,
                timestamp: new Date().toISOString()
            };
            
            performanceMetrics.pageLoads.push(pageLoadMetric);
            
            console.log('📊 Page Load Performance:', pageLoadMetric);
            
            if (pageLoadMetric.totalLoadTime > 3000) {
                console.warn('🐌 Slow page load detected:', pageLoadMetric.totalLoadTime + 'ms');
            }
            
            // Core Web Vitals
            measureCoreWebVitals();
        });
    }
    
    /**
     * Measure Core Web Vitals
     */
    function measureCoreWebVitals() {
        // Largest Contentful Paint (LCP)
        if (window.PerformanceObserver) {
            new PerformanceObserver((entryList) => {
                const entries = entryList.getEntries();
                const lastEntry = entries[entries.length - 1];
                const lcp = lastEntry.startTime;
                
                console.log(`📈 LCP (Largest Contentful Paint): ${lcp.toFixed(2)}ms`);
                
                if (lcp > 2500) {
                    console.warn('🐌 Poor LCP detected:', lcp + 'ms');
                } else if (lcp > 1500) {
                    console.log('⚠️ Needs improvement LCP:', lcp + 'ms');
                } else {
                    console.log('✅ Good LCP:', lcp + 'ms');
                }
            }).observe({ entryTypes: ['largest-contentful-paint'] });
            
            // First Input Delay (FID)
            new PerformanceObserver((entryList) => {
                entryList.getEntries().forEach((entry) => {
                    const fid = entry.processingStart - entry.startTime;
                    console.log(`⚡ FID (First Input Delay): ${fid.toFixed(2)}ms`);
                    
                    if (fid > 100) {
                        console.warn('🐌 Poor FID detected:', fid + 'ms');
                    } else if (fid > 50) {
                        console.log('⚠️ Needs improvement FID:', fid + 'ms');
                    } else {
                        console.log('✅ Good FID:', fid + 'ms');
                    }
                });
            }).observe({ entryTypes: ['first-input'] });
            
            // Cumulative Layout Shift (CLS)
            let clsValue = 0;
            new PerformanceObserver((entryList) => {
                entryList.getEntries().forEach((entry) => {
                    if (!entry.hadRecentInput) {
                        clsValue += entry.value;
                    }
                });
                
                console.log(`📐 CLS (Cumulative Layout Shift): ${clsValue.toFixed(4)}`);
                
                if (clsValue > 0.25) {
                    console.warn('🐌 Poor CLS detected:', clsValue);
                } else if (clsValue > 0.1) {
                    console.log('⚠️ Needs improvement CLS:', clsValue);
                } else {
                    console.log('✅ Good CLS:', clsValue);
                }
            }).observe({ entryTypes: ['layout-shift'] });
        }
    }
    
    /**
     * Measure user interaction performance
     */
    function measureUserInteractions() {
        if (!config.measureUserInteractions) return;
        
        // Track click performance
        document.addEventListener('click', function(event) {
            const startTime = performance.now();
            
            setTimeout(() => {
                const endTime = performance.now();
                const duration = endTime - startTime;
                
                const interactionMetric = {
                    type: 'click',
                    target: event.target.tagName + (event.target.id ? '#' + event.target.id : ''),
                    duration: duration.toFixed(2),
                    timestamp: new Date().toISOString()
                };
                
                performanceMetrics.userInteractions.push(interactionMetric);
                
                if (duration > config.logThreshold) {
                    console.warn('🐌 Slow interaction detected:', interactionMetric);
                }
            }, 0);
        });
        
        // Track form submission performance
        document.addEventListener('submit', function(event) {
            const startTime = performance.now();
            const form = event.target;
            
            setTimeout(() => {
                const endTime = performance.now();
                const duration = endTime - startTime;
                
                console.log(`📝 Form submission performance: ${duration.toFixed(2)}ms`);
            }, 0);
        });
    }
    
    /**
     * Monitor resource loading performance
     */
    function monitorResourceLoading() {
        if (window.PerformanceObserver) {
            new PerformanceObserver((entryList) => {
                entryList.getEntries().forEach((entry) => {
                    const resourceMetric = {
                        name: entry.name,
                        type: entry.initiatorType,
                        duration: entry.duration.toFixed(2),
                        size: entry.transferSize || entry.encodedBodySize || 0,
                        cached: entry.transferSize === 0 && entry.encodedBodySize > 0,
                        timestamp: new Date().toISOString()
                    };
                    
                    performanceMetrics.resourceLoading.push(resourceMetric);
                    
                    if (entry.duration > config.logThreshold) {
                        console.warn('🐌 Slow resource loading:', resourceMetric);
                    }
                });
            }).observe({ entryTypes: ['resource'] });
        }
    }
    
    /**
     * Generate performance report
     */
    function generatePerformanceReport() {
        console.group('📊 Performance Report Summary');
        console.log('API Calls:', performanceMetrics.apiCalls.length);
        console.log('Page Loads:', performanceMetrics.pageLoads.length);
        console.log('User Interactions:', performanceMetrics.userInteractions.length);
        console.log('Resource Loads:', performanceMetrics.resourceLoading.length);
        
        // Calculate averages
        if (performanceMetrics.apiCalls.length > 0) {
            const avgApiTime = performanceMetrics.apiCalls.reduce((sum, call) => sum + parseFloat(call.duration), 0) / performanceMetrics.apiCalls.length;
            console.log('Average API Call Time:', avgApiTime.toFixed(2) + 'ms');
        }
        
        if (performanceMetrics.userInteractions.length > 0) {
            const avgInteractionTime = performanceMetrics.userInteractions.reduce((sum, interaction) => sum + parseFloat(interaction.duration), 0) / performanceMetrics.userInteractions.length;
            console.log('Average Interaction Time:', avgInteractionTime.toFixed(2) + 'ms');
        }
        
        console.groupEnd();
        
        // Return full metrics for external analysis
        return performanceMetrics;
    }
    
    /**
     * Start memory monitoring
     */
    function startMemoryMonitoring() {
        if (window.performance && window.performance.memory) {
            setInterval(() => {
                const memory = window.performance.memory;
                const memoryInfo = {
                    usedJSHeapSize: (memory.usedJSHeapSize / 1048576).toFixed(2) + ' MB',
                    totalJSHeapSize: (memory.totalJSHeapSize / 1048576).toFixed(2) + ' MB',
                    jsHeapSizeLimit: (memory.jsHeapSizeLimit / 1048576).toFixed(2) + ' MB',
                    timestamp: new Date().toISOString()
                };
                
                // Log memory warning if usage is high
                const usagePercent = (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100;
                if (usagePercent > 80) {
                    console.warn('🔴 High memory usage detected:', memoryInfo);
                } else if (usagePercent > 60) {
                    console.log('🟡 Moderate memory usage:', memoryInfo);
                }
            }, 10000); // Check every 10 seconds
        }
    }
    
    // Initialize performance monitoring
    measureApiPerformance();
    measurePageLoadPerformance();
    measureUserInteractions();
    monitorResourceLoading();
    startMemoryMonitoring();
    
    // Expose global functions for manual testing
    window.performanceMonitor = {
        getMetrics: () => performanceMetrics,
        generateReport: generatePerformanceReport,
        logThreshold: config.logThreshold,
        config: config
    };
    
    console.log('✅ Performance monitoring initialized');
    console.log('💡 Use window.performanceMonitor.generateReport() to view summary');
    console.log('💡 Use window.performanceMonitor.getMetrics() to view raw data');
    
})();

/**
 * Usage Examples:
 * 
 * // Generate performance report
 * window.performanceMonitor.generateReport();
 * 
 * // Get specific metrics
 * window.performanceMonitor.getMetrics().apiCalls;
 * 
 * // Adjust logging threshold
 * window.performanceMonitor.logThreshold = 50; // Log operations slower than 50ms
 * 
 * // Manual performance mark
 * performance.mark('operation-start');
 * // ... your code ...
 * performance.mark('operation-end');
 * performance.measure('operation-duration', 'operation-start', 'operation-end');
 * console.log(performance.getEntriesByName('operation-duration'));
 */
