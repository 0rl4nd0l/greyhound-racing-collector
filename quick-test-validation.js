#!/usr/bin/env node

// Quick validation test for helper routes
// This demonstrates the test infrastructure without running full browser tests

const http = require('http');
const { spawn } = require('child_process');

const PORT = 5678;
const HOST = 'localhost';

let flaskProcess = null;

// Colors for output
const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    blue: '\x1b[34m',
    yellow: '\x1b[33m'
};

function log(message, color = colors.reset) {
    console.log(`${color}${message}${colors.reset}`);
}

function makeRequest(path) {
    return new Promise((resolve, reject) => {
        const options = {
            hostname: HOST,
            port: PORT,
            path: path,
            method: 'GET',
            timeout: 5000
        };

        const req = http.request(options, (res) => {
            let data = '';
            res.on('data', (chunk) => data += chunk);
            res.on('end', () => {
                resolve({
                    statusCode: res.statusCode,
                    headers: res.headers,
                    body: data
                });
            });
        });

        req.on('error', reject);
        req.on('timeout', () => {
            req.destroy();
            reject(new Error('Request timeout'));
        });
        
        req.end();
    });
}

function startFlaskApp() {
    return new Promise((resolve, reject) => {
        log('🚀 Starting Flask app in testing mode...', colors.blue);
        
        const env = {
            ...process.env,
            TESTING: 'true',
            FLASK_ENV: 'testing',
            MODULE_GUARD_STRICT: '0',
            PREDICTION_IMPORT_MODE: 'relaxed'
        };

        flaskProcess = spawn('python', ['app.py', '--host', HOST, '--port', PORT.toString()], {
            env: env,
            stdio: ['ignore', 'pipe', 'pipe']
        });

        flaskProcess.stdout.on('data', (data) => {
            if (data.toString().includes('Running on')) {
                log('✅ Flask app is running', colors.green);
                setTimeout(resolve, 2000); // Give it a moment to fully start
            }
        });

        flaskProcess.stderr.on('data', (data) => {
            // Suppress most stderr output, but show errors
            const output = data.toString();
            if (output.includes('ERROR') || output.includes('CRITICAL')) {
                log(`Flask Error: ${output.trim()}`, colors.red);
            }
        });

        flaskProcess.on('error', reject);
        
        // Timeout after 10 seconds
        setTimeout(() => {
            if (!flaskProcess.killed) {
                resolve(); // Proceed anyway
            }
        }, 10000);
    });
}

function stopFlaskApp() {
    if (flaskProcess && !flaskProcess.killed) {
        log('🛑 Stopping Flask app...', colors.yellow);
        flaskProcess.kill();
    }
}

async function testHelperRoutes() {
    log('=== Helper Routes Validation Test ===', colors.blue);
    
    const routes = [
        { path: '/ping', description: 'Health check endpoint' },
        { path: '/test-blank-page', description: 'Blank page for script injection' },
        { path: '/test-predictions', description: 'Predictions testing page' },
        { path: '/test-sidebar', description: 'Sidebar layout testing page' }
    ];

    for (const route of routes) {
        try {
            log(`\n📡 Testing ${route.path} - ${route.description}`, colors.blue);
            const response = await makeRequest(route.path);
            
            if (response.statusCode === 200) {
                log(`✅ ${route.path} -> HTTP ${response.statusCode}`, colors.green);
                
                // Basic content validation
                if (route.path === '/test-blank-page' && response.body.includes('test-container')) {
                    log('  ✓ Contains test-container element', colors.green);
                } else if (route.path === '/test-predictions' && response.body.includes('predictions-results-container')) {
                    log('  ✓ Contains predictions-results-container element', colors.green);
                } else if (route.path === '/test-sidebar' && response.body.includes('sidebar-logs')) {
                    log('  ✓ Contains sidebar-logs element', colors.green);
                } else if (route.path === '/ping') {
                    log('  ✓ Ping successful', colors.green);
                }
                
                // Check for Bootstrap CSS
                if (response.body.includes('bootstrap') || response.body.includes('container-fluid')) {
                    log('  ✓ Bootstrap CSS detected', colors.green);
                }
                
            } else {
                log(`❌ ${route.path} -> HTTP ${response.statusCode}`, colors.red);
            }
            
        } catch (error) {
            log(`❌ ${route.path} -> Error: ${error.message}`, colors.red);
        }
    }
}

async function main() {
    try {
        await startFlaskApp();
        await testHelperRoutes();
        
        log('\n🎉 Helper routes validation completed!', colors.green);
        log('\n📝 Next Steps:', colors.blue);
        log('• Run full Cypress tests: ./run-tests.sh -t cypress', colors.reset);
        log('• Run full Playwright tests: ./run-tests.sh -t playwright', colors.reset);
        log('• Run all tests: ./run-tests.sh', colors.reset);
        
    } catch (error) {
        log(`\n💥 Error: ${error.message}`, colors.red);
        process.exit(1);
    } finally {
        stopFlaskApp();
        setTimeout(() => process.exit(0), 1000);
    }
}

// Handle interruption
process.on('SIGINT', () => {
    log('\n🛑 Interrupted by user', colors.yellow);
    stopFlaskApp();
    process.exit(0);
});

process.on('SIGTERM', () => {
    stopFlaskApp();
    process.exit(0);
});

main();
