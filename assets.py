"""
Asset Management System for Greyhound Racing Dashboard
=====================================================

This module configures Flask-Assets for concatenating, minifying, and cache-busting
CSS and JavaScript files.
"""

import os
import hashlib
from flask_assets import Bundle, Environment
from flask_compress import Compress

def init_assets(app):
    """Initialize asset management and compression for the Flask app."""
    
    # Initialize Flask-Assets
    assets = Environment(app)
    
    # Configure asset directories
    assets.directory = app.static_folder
    assets.url = app.static_url_path
    
    # Enable debugging in development
    assets.debug = app.debug
    assets.auto_build = True
    
    # Configure output directory for bundled assets
    assets.config['ASSETS_DIR'] = os.path.join(app.static_folder, 'dist')
    assets.config['ASSETS_URL'] = '/static/dist/'
    
    # Ensure dist directories exist
    os.makedirs(os.path.join(app.static_folder, 'dist', 'css'), exist_ok=True)
    os.makedirs(os.path.join(app.static_folder, 'dist', 'js'), exist_ok=True)
    
    # CSS Bundle - concatenate all CSS files
    css_bundle = Bundle(
        'css/variables.css',
        'css/style.css',
        'css/components.css',
        'css/utilities.css',
        'css/ml-dashboard.css',
        'css/interactive-races.css',
        filters='cssmin,autoprefixer',
        output='dist/css/bundle-%(version)s.css'
    )
    
    # JavaScript Bundle - concatenate core JS files
    js_bundle = Bundle(
        'js/script.js',
        'js/sidebar.js',
        'js/loading-utils.js',
        'js/prediction-buttons.js',
        filters='jsmin',
        output='dist/js/bundle-%(version)s.js'
    )
    
    # Dashboard-specific JS Bundle
    dashboard_js_bundle = Bundle(
        'js/ml-dashboard.js',
        'js/model-registry.js',
        'js/monitoring.js',
        filters='jsmin',
        output='dist/js/dashboard-%(version)s.js'
    )
    
    # Interactive features JS Bundle
    interactive_js_bundle = Bundle(
        'js/interactive-races.js',
        'js/dogs_analysis.js',
        'js/predictions_v2.js',
        filters='jsmin',
        output='dist/js/interactive-%(version)s.js'
    )
    
    # Utils JS Bundle - Standalone utilities
    utils_js_bundle = Bundle(
        'js/advisoryUtils.js',
        'js/null-safe-sorting-example.js',
        filters='jsmin',
        output='dist/js/utils-%(version)s.js'
    )
    
    # Register bundles
    assets.register('css_bundle', css_bundle)
    assets.register('js_bundle', js_bundle)
    assets.register('dashboard_js', dashboard_js_bundle)
    assets.register('interactive_js', interactive_js_bundle)
    assets.register('utils_js', utils_js_bundle)
    
    # Initialize Flask-Compress for gzip/brotli compression
    compress = Compress()
    compress.init_app(app)
    
    # Configure compression settings
    app.config['COMPRESS_MIMETYPES'] = [
        'text/html',
        'text/css',
        'text/xml',
        'application/json',
        'application/javascript',
        'application/xml+rss',
        'application/atom+xml',
        'text/javascript',
        'image/svg+xml'
    ]
    
    app.config['COMPRESS_LEVEL'] = 6
    app.config['COMPRESS_MIN_SIZE'] = 500
    app.config['COMPRESS_ALGORITHM'] = 'gzip'
    
    return assets, compress


def get_asset_hash(content):
    """Generate a hash for cache busting."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:8]


def create_main_css():
    """Create main.css file that imports all component CSS files."""
    main_css_content = """/* Main CSS file - imports all components */
@import 'variables.css';
@import 'style.css';
@import 'components.css';
@import 'utilities.css';
@import 'ml-dashboard.css';
@import 'interactive-races.css';
"""
    
    css_path = os.path.join('static', 'css', 'main.css')
    os.makedirs(os.path.dirname(css_path), exist_ok=True)
    
    with open(css_path, 'w') as f:
        f.write(main_css_content)
    
    return css_path


def create_main_js():
    """Create main.js file that imports all core JavaScript files."""
    main_js_content = """// Main JavaScript file - imports all core components
// Core utilities and global functions
(function() {
    'use strict';
    
    // Import core functionality
    // This would be replaced by a proper bundling system in production
    
    console.log('Greyhound Racing Dashboard - Asset bundle loaded');
    
    // Initialize global features
    if (typeof window.initializeTooltips === 'function') {
        window.initializeTooltips();
    }
    
    if (typeof window.initializeTheme === 'function') {
        window.initializeTheme();
    }
    
})();
"""
    
    js_path = os.path.join('static', 'js', 'main.js')
    os.makedirs(os.path.dirname(js_path), exist_ok=True)
    
    with open(js_path, 'w') as f:
        f.write(main_js_content)
    
    return js_path


class AssetManager:
    """Utility class for managing assets and cache busting."""
    
    def __init__(self, app=None):
        self.app = app
        self.assets = None
        self.compress = None
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the asset manager with the Flask app."""
        self.app = app
        self.assets, self.compress = init_assets(app)
        
        # Create main asset files if they don't exist
        create_main_css()
        create_main_js()
        
        # Add template globals for asset URLs
        app.jinja_env.globals['asset_url'] = self.asset_url
        app.jinja_env.globals['css_bundle'] = self.css_bundle_url
        app.jinja_env.globals['js_bundle'] = self.js_bundle_url
    
    def asset_url(self, filename):
        """Get the URL for a static asset with cache busting."""
        if self.assets:
            return self.assets.url + filename
        return f"/static/{filename}"
    
    def css_bundle_url(self):
        """Get the URL for the CSS bundle."""
        if self.assets and 'css_bundle' in self.assets:
            return self.assets['css_bundle'].urls()[0]
        return "/static/dist/css/bundle.css"
    
    def js_bundle_url(self):
        """Get the URL for the JS bundle."""
        if self.assets and 'js_bundle' in self.assets:
            return self.assets['js_bundle'].urls()[0]
        return "/static/dist/js/bundle.js"
