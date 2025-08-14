#!/usr/bin/env node

/**
 * Automated Performance Profiler using Puppeteer
 * 
 * This script automates the capture of performance metrics including:
 * - Full page-load waterfall
 * - FPS measurements
 * - First-byte, DOMContentLoaded, full load times
 * - Network metrics (requests, transfer sizes)
 * - Time-to-Interactive
 * - HAR files and Lighthouse reports
 */

const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const path = require('path');

const DASHBOARD_URL = 'http://localhost:5002';
const REPORTS_DIR = './perf_reports/baseline/';
const PROFILE_DURATION = 15000; // 15 seconds

class AutomatedPerformanceProfiler {
    constructor() {
        this.browser = null;
        this.page = null;
        this.metrics = {};
        this.harData = null;
        this.lighthouseReport = null;
        this.timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    }

    /**
     * Initialize browser and page
     */
    async initialize() {
        console.log('🚀 Initializing Automated Performance Profiler...');
        
        // Ensure reports directory exists
        await fs.mkdir(REPORTS_DIR, { recursive: true });
        
        // Launch browser with performance flags
        this.browser = await puppeteer.launch({
            headless: false,
            devtools: true,
            args: [
                '--enable-precise-memory-info',
                '--enable-gpu-benchmarking',
                '--enable-thread-ed-compositing',
                '--disable-background-timer-throttling',
                '--disable-renderer-backgrounding',
                '--disable-backgrounding-occluded-windows',
                '--no-sandbox',
                '--disable-web-security'
            ]
        });
        
        this.page = await this.browser.newPage();
        
        // Set viewport for consistent testing
        await this.page.setViewport({ width: 1920, height: 1080 });
        
        // Enable performance tracking
        await this.page.tracing.start({
            path: path.join(REPORTS_DIR, `trace-${this.timestamp}.json`),
            screenshots: true,
            categories: ['devtools.timeline', 'disabled-by-default-devtools.timeline']
        });
        
        // Enable network domain
        const client = await this.page.target().createCDPSession();
        await client.send('Network.enable');
        await client.send('Performance.enable');
        
        console.log('✅ Browser and profiling initialized');
        return client;
    }

    /**
     * Run comprehensive performance profiling
     */
    async runProfiling() {
        console.log('🔍 Starting comprehensive performance profiling...');
        
        const client = await this.initialize();
        
        // Start performance monitoring
        const performanceMetrics = [];
        const networkRequests = [];
        let harEntries = [];
        
        // Collect network requests
        client.on('Network.responseReceived', (event) => {
            networkRequests.push(event);
        });
        
        // Start navigation timing
        const navigationStart = Date.now();
        console.log(`📄 Loading ${DASHBOARD_URL}...`);
        
        // Navigate to dashboard with network idle wait
        const response = await this.page.goto(DASHBOARD_URL, {
            waitUntil: ['networkidle0', 'domcontentloaded'],
            timeout: 30000
        });
        
        console.log(`✅ Initial page load complete - Status: ${response.status()}`);
        
        // Wait for additional resources and JavaScript execution
        await this.page.waitForTimeout(3000);
        
        // Capture performance metrics
        const perfMetrics = await this.capturePerformanceMetrics();
        const navigationMetrics = await this.captureNavigationTiming();
        const networkMetrics = await this.captureNetworkMetrics();
        
        // Generate FPS measurements by monitoring for a period
        console.log('📊 Measuring FPS over 10 seconds...');
        const fpsMetrics = await this.measureFPS(10000);
        
        // Capture screenshots at key moments
        await this.captureScreenshots();
        
        // Generate HAR data
        await this.generateHAR(networkRequests);
        
        // Create Lighthouse-style report
        this.createLighthouseReport({
            ...perfMetrics,
            ...navigationMetrics,
            ...networkMetrics,
            ...fpsMetrics
        });
        
        // Stop tracing
        await this.page.tracing.stop();
        
        // Save all reports
        await this.saveReports();
        
        // Display results
        this.displayResults();
        
        await this.browser.close();
        console.log('🎯 Performance profiling completed successfully!');
    }

    /**
     * Capture performance metrics using Chrome DevTools Protocol
     */
    async capturePerformanceMetrics() {
        console.log('📈 Capturing performance metrics...');
        
        const metrics = await this.page.evaluate(() => {
            const navigation = performance.getEntriesByType('navigation')[0];
            const paint = performance.getEntriesByType('paint');
            
            const result = {
                // Navigation timing
                navigationStart: navigation.startTime,
                fetchStart: navigation.fetchStart,
                domainLookupStart: navigation.domainLookupStart,
                domainLookupEnd: navigation.domainLookupEnd,
                connectStart: navigation.connectStart,
                connectEnd: navigation.connectEnd,
                requestStart: navigation.requestStart,
                responseStart: navigation.responseStart,
                responseEnd: navigation.responseEnd,
                domLoading: navigation.domContentLoadedEventStart,
                domContentLoaded: navigation.domContentLoadedEventEnd,
                domComplete: navigation.domComplete,
                loadComplete: navigation.loadEventEnd,
                
                // Calculated metrics
                firstByte: navigation.responseStart - navigation.fetchStart,
                domContentLoadedTime: navigation.domContentLoadedEventEnd - navigation.fetchStart,
                fullLoadTime: navigation.loadEventEnd - navigation.fetchStart,
                timeToInteractive: navigation.domComplete - navigation.fetchStart
            };
            
            // Paint timing
            paint.forEach(entry => {
                if (entry.name === 'first-paint') {
                    result.firstPaint = entry.startTime;
                } else if (entry.name === 'first-contentful-paint') {
                    result.firstContentfulPaint = entry.startTime;
                }
            });
            
            // Largest Contentful Paint
            const lcpEntries = performance.getEntriesByType('largest-contentful-paint');
            if (lcpEntries.length > 0) {
                result.largestContentfulPaint = lcpEntries[lcpEntries.length - 1].startTime;
            }
            
            return result;
        });
        
        this.metrics = { ...this.metrics, ...metrics };
        return metrics;
    }

    /**
     * Capture navigation timing details
     */
    async captureNavigationTiming() {
        console.log('🧭 Capturing navigation timing...');
        
        const navigation = await this.page.evaluate(() => {
            const nav = performance.getEntriesByType('navigation')[0];
            return {
                redirectCount: nav.redirectCount,
                type: nav.type,
                transferSize: nav.transferSize,
                encodedBodySize: nav.encodedBodySize,
                decodedBodySize: nav.decodedBodySize
            };
        });
        
        return navigation;
    }

    /**
     * Capture network metrics
     */
    async captureNetworkMetrics() {
        console.log('🌐 Capturing network metrics...');
        
        const networkData = await this.page.evaluate(() => {
            const resources = performance.getEntriesByType('resource');
            let totalRequests = resources.length;
            let totalTransferred = 0;
            let totalDecoded = 0;
            
            const resourceTypes = {};
            const slowestResources = [];
            
            resources.forEach(resource => {
                // Calculate total transfer
                if (resource.transferSize) {
                    totalTransferred += resource.transferSize;
                }
                if (resource.decodedBodySize) {
                    totalDecoded += resource.decodedBodySize;
                }
                
                // Categorize by type
                const type = resource.name.includes('.css') ? 'CSS' :
                           resource.name.includes('.js') ? 'JavaScript' :
                           resource.name.match(/\.(png|jpg|jpeg|gif|svg|webp)$/i) ? 'Image' :
                           resource.name.includes('/api/') ? 'API' : 'Other';
                
                if (!resourceTypes[type]) {
                    resourceTypes[type] = { count: 0, size: 0, duration: 0 };
                }
                resourceTypes[type].count++;
                if (resource.transferSize) {
                    resourceTypes[type].size += resource.transferSize;
                }
                resourceTypes[type].duration += (resource.responseEnd - resource.startTime);
                
                // Track slowest resources
                const duration = resource.responseEnd - resource.startTime;
                slowestResources.push({
                    name: resource.name.split('/').pop() || resource.name,
                    duration: duration,
                    size: resource.transferSize || 0,
                    type: type
                });
            });
            
            // Sort slowest resources
            slowestResources.sort((a, b) => b.duration - a.duration);
            
            return {
                totalRequests,
                totalTransferred,
                totalDecoded,
                resourceTypes,
                slowestResources: slowestResources.slice(0, 10) // Top 10 slowest
            };
        });
        
        this.metrics = { ...this.metrics, ...networkData };
        return networkData;
    }

    /**
     * Measure FPS over a given duration
     */
    async measureFPS(duration) {
        console.log(`🎬 Measuring FPS for ${duration/1000} seconds...`);
        
        const fpsData = await this.page.evaluate((measureDuration) => {
            return new Promise((resolve) => {
                const frames = [];
                let startTime = performance.now();
                let frameCount = 0;
                
                function measureFrame() {
                    const currentTime = performance.now();
                    frameCount++;
                    
                    if (currentTime - startTime >= measureDuration) {
                        const fps = (frameCount * 1000) / (currentTime - startTime);
                        resolve({
                            averageFPS: fps,
                            frameCount: frameCount,
                            duration: currentTime - startTime
                        });
                    } else {
                        requestAnimationFrame(measureFrame);
                    }
                }
                
                requestAnimationFrame(measureFrame);
            });
        }, duration);
        
        return fpsData;
    }

    /**
     * Capture screenshots at key moments
     */
    async captureScreenshots() {
        console.log('📸 Capturing screenshots...');
        
        // Full page screenshot
        await this.page.screenshot({
            path: path.join(REPORTS_DIR, `screenshot-full-${this.timestamp}.png`),
            fullPage: true
        });
        
        // Viewport screenshot
        await this.page.screenshot({
            path: path.join(REPORTS_DIR, `screenshot-viewport-${this.timestamp}.png`)
        });
    }

    /**
     * Generate HAR (HTTP Archive) data
     */
    async generateHAR(networkRequests) {
        console.log('📋 Generating HAR data...');
        
        const harEntries = networkRequests.map(request => ({
            startedDateTime: new Date().toISOString(),
            time: 0, // This would need to be calculated from timing data
            request: {
                method: 'GET',
                url: request.response.url,
                httpVersion: request.response.protocol || 'HTTP/1.1',
                headers: Object.entries(request.response.headers || {}).map(([name, value]) => ({name, value}))
            },
            response: {
                status: request.response.status,
                statusText: request.response.statusText,
                httpVersion: request.response.protocol || 'HTTP/1.1',
                headers: Object.entries(request.response.headers || {}).map(([name, value]) => ({name, value}))
            }
        }));
        
        this.harData = {
            log: {
                version: '1.2',
                creator: {
                    name: 'Automated Performance Profiler',
                    version: '1.0'
                },
                entries: harEntries
            }
        };
    }

    /**
     * Create Lighthouse-style performance report
     */
    createLighthouseReport(allMetrics) {
        console.log('🏆 Creating Lighthouse-style report...');
        
        this.lighthouseReport = {
            timestamp: Date.now(),
            url: DASHBOARD_URL,
            metrics: {
                'first-contentful-paint': {
                    value: allMetrics.firstContentfulPaint || 0,
                    score: this.scoreMetric(allMetrics.firstContentfulPaint || 0, [1000, 2000]),
                    unit: 'ms'
                },
                'largest-contentful-paint': {
                    value: allMetrics.largestContentfulPaint || 0,
                    score: this.scoreMetric(allMetrics.largestContentfulPaint || 0, [2500, 4000]),
                    unit: 'ms'
                },
                'time-to-interactive': {
                    value: allMetrics.timeToInteractive || 0,
                    score: this.scoreMetric(allMetrics.timeToInteractive || 0, [3000, 6000]),
                    unit: 'ms'
                },
                'first-byte': {
                    value: allMetrics.firstByte || 0,
                    score: this.scoreMetric(allMetrics.firstByte || 0, [200, 500]),
                    unit: 'ms'
                },
                'dom-content-loaded': {
                    value: allMetrics.domContentLoadedTime || 0,
                    score: this.scoreMetric(allMetrics.domContentLoadedTime || 0, [1500, 3000]),
                    unit: 'ms'
                },
                'full-load': {
                    value: allMetrics.fullLoadTime || 0,
                    score: this.scoreMetric(allMetrics.fullLoadTime || 0, [3000, 6000]),
                    unit: 'ms'
                }
            },
            performance: {
                score: 0 // Will be calculated
            },
            network: {
                totalRequests: allMetrics.totalRequests || 0,
                totalTransferred: allMetrics.totalTransferred || 0,
                totalDecoded: allMetrics.totalDecoded || 0,
                resourceBreakdown: allMetrics.resourceTypes || {},
                slowestResources: allMetrics.slowestResources || []
            },
            fps: {
                average: allMetrics.averageFPS || 0,
                frameCount: allMetrics.frameCount || 0
            }
        };
        
        // Calculate overall performance score
        const metricScores = Object.values(this.lighthouseReport.metrics).map(m => m.score);
        this.lighthouseReport.performance.score = Math.round(
            metricScores.reduce((a, b) => a + b, 0) / metricScores.length
        );
    }

    /**
     * Score a metric based on thresholds (0-100 scale)
     */
    scoreMetric(value, thresholds) {
        const [good, poor] = thresholds;
        if (value <= good) return 100;
        if (value >= poor) return 0;
        return Math.round(100 * (poor - value) / (poor - good));
    }

    /**
     * Save all reports to files
     */
    async saveReports() {
        console.log('💾 Saving performance reports...');
        
        try {
            // Save HAR file
            if (this.harData) {
                await fs.writeFile(
                    path.join(REPORTS_DIR, `baseline-har-${this.timestamp}.json`),
                    JSON.stringify(this.harData, null, 2)
                );
            }
            
            // Save Lighthouse report
            if (this.lighthouseReport) {
                await fs.writeFile(
                    path.join(REPORTS_DIR, `baseline-lighthouse-${this.timestamp}.json`),
                    JSON.stringify(this.lighthouseReport, null, 2)
                );
            }
            
            // Save raw metrics
            await fs.writeFile(
                path.join(REPORTS_DIR, `baseline-metrics-${this.timestamp}.json`),
                JSON.stringify(this.metrics, null, 2)
            );
            
            // Save summary report
            const summary = this.generateSummaryReport();
            await fs.writeFile(
                path.join(REPORTS_DIR, `baseline-summary-${this.timestamp}.md`),
                summary
            );
            
            console.log(`✅ Reports saved to ${REPORTS_DIR}`);
        } catch (error) {
            console.error('❌ Error saving reports:', error);
        }
    }

    /**
     * Generate human-readable summary report
     */
    generateSummaryReport() {
        const report = this.lighthouseReport;
        
        return `# Performance Baseline Report

**Generated:** ${new Date().toISOString()}
**URL:** ${DASHBOARD_URL}
**Overall Performance Score:** ${report.performance.score}/100

## Core Web Vitals

| Metric | Value | Score | Status |
|--------|-------|-------|---------|
| First Contentful Paint | ${report.metrics['first-contentful-paint'].value.toFixed(2)}ms | ${report.metrics['first-contentful-paint'].score}/100 | ${report.metrics['first-contentful-paint'].score >= 75 ? '✅ Good' : report.metrics['first-contentful-paint'].score >= 50 ? '⚠️ Needs Improvement' : '❌ Poor'} |
| Largest Contentful Paint | ${report.metrics['largest-contentful-paint'].value.toFixed(2)}ms | ${report.metrics['largest-contentful-paint'].score}/100 | ${report.metrics['largest-contentful-paint'].score >= 75 ? '✅ Good' : report.metrics['largest-contentful-paint'].score >= 50 ? '⚠️ Needs Improvement' : '❌ Poor'} |
| Time to Interactive | ${report.metrics['time-to-interactive'].value.toFixed(2)}ms | ${report.metrics['time-to-interactive'].score}/100 | ${report.metrics['time-to-interactive'].score >= 75 ? '✅ Good' : report.metrics['time-to-interactive'].score >= 50 ? '⚠️ Needs Improvement' : '❌ Poor'} |

## Loading Performance

| Metric | Value | Score |
|--------|-------|-------|
| First Byte | ${report.metrics['first-byte'].value.toFixed(2)}ms | ${report.metrics['first-byte'].score}/100 |
| DOMContentLoaded | ${report.metrics['dom-content-loaded'].value.toFixed(2)}ms | ${report.metrics['dom-content-loaded'].score}/100 |
| Full Load | ${report.metrics['full-load'].value.toFixed(2)}ms | ${report.metrics['full-load'].score}/100 |

## Network Analysis

- **Total Requests:** ${report.network.totalRequests}
- **Total Transferred:** ${(report.network.totalTransferred / 1024).toFixed(2)} KB
- **Total Decoded:** ${(report.network.totalDecoded / 1024).toFixed(2)} KB
- **Average FPS:** ${report.fps.average.toFixed(2)}

## Resource Breakdown

${Object.entries(report.network.resourceBreakdown).map(([type, data]) => 
    `- **${type}:** ${data.count} files, ${(data.size / 1024).toFixed(2)} KB`
).join('\n')}

## Slowest Resources

${report.network.slowestResources.slice(0, 5).map((resource, index) => 
    `${index + 1}. **${resource.name}** (${resource.type}): ${resource.duration.toFixed(2)}ms, ${(resource.size / 1024).toFixed(2)} KB`
).join('\n')}

## Recommendations

${this.generateRecommendations()}
`;
    }

    /**
     * Generate performance recommendations
     */
    generateRecommendations() {
        const recommendations = [];
        const report = this.lighthouseReport;
        
        if (report.metrics['first-contentful-paint'].score < 75) {
            recommendations.push('- **Improve First Contentful Paint:** Optimize critical resources, reduce server response time, eliminate render-blocking resources');
        }
        
        if (report.metrics['largest-contentful-paint'].score < 75) {
            recommendations.push('- **Improve Largest Contentful Paint:** Optimize images, preload important resources, improve server response time');
        }
        
        if (report.network.totalRequests > 100) {
            recommendations.push('- **Reduce Network Requests:** Combine files, use CSS sprites, implement lazy loading');
        }
        
        if (report.network.totalTransferred > 1024 * 1024) { // > 1MB
            recommendations.push('- **Reduce Transfer Size:** Enable compression, optimize images, minify resources');
        }
        
        if (report.fps.average < 30) {
            recommendations.push('- **Improve Frame Rate:** Optimize JavaScript execution, reduce paint complexity, use CSS transforms');
        }
        
        return recommendations.length > 0 ? recommendations.join('\n') : '✅ Performance looks good! Continue monitoring and optimizing.';
    }

    /**
     * Display results in console
     */
    displayResults() {
        console.log('\n📊 Performance Profiling Results Summary:');
        console.log('==========================================');
        
        const report = this.lighthouseReport;
        
        console.log(`🏆 Overall Performance Score: ${report.performance.score}/100`);
        console.log(`🚀 First Byte: ${report.metrics['first-byte'].value.toFixed(2)}ms`);
        console.log(`📄 DOMContentLoaded: ${report.metrics['dom-content-loaded'].value.toFixed(2)}ms`);
        console.log(`✅ Full Load: ${report.metrics['full-load'].value.toFixed(2)}ms`);
        console.log(`🎯 Time to Interactive: ${report.metrics['time-to-interactive'].value.toFixed(2)}ms`);
        console.log(`🎨 First Contentful Paint: ${report.metrics['first-contentful-paint'].value.toFixed(2)}ms`);
        console.log(`📦 Total Requests: ${report.network.totalRequests}`);
        console.log(`📊 Total Transferred: ${(report.network.totalTransferred / 1024).toFixed(2)}KB`);
        console.log(`🎬 Average FPS: ${report.fps.average.toFixed(2)}`);
        
        console.log('\n📋 Resource Breakdown:');
        Object.entries(report.network.resourceBreakdown).forEach(([type, data]) => {
            console.log(`  ${type}: ${data.count} files, ${(data.size / 1024).toFixed(2)}KB`);
        });
    }
}

// Run profiling if called directly
if (require.main === module) {
    const profiler = new AutomatedPerformanceProfiler();
    profiler.runProfiling().catch(console.error);
}

module.exports = AutomatedPerformanceProfiler;
