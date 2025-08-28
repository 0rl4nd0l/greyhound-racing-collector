import { defineConfig } from 'vite';
import { resolve } from 'path';

// Vite config scoped to the static/ directory
// Emits hashed assets to static/dist and a manifest.json for server-side resolution
export default defineConfig({
  root: resolve(__dirname),
  base: '/static/',
  build: {
    outDir: resolve(__dirname, 'dist'),
    emptyOutDir: true,
    manifest: true,
    target: 'es2017',
    cssCodeSplit: true,
    rollupOptions: {
      input: {
        app: resolve(__dirname, 'js/main.js'),
        interactive: resolve(__dirname, 'js/interactive-races.js'),
        predictionButtons: resolve(__dirname, 'js/prediction-buttons.js'),
        monitoring: resolve(__dirname, 'js/monitoring.js'),
        mlDashboard: resolve(__dirname, 'js/ml-dashboard.js'),
        // Keep underscore variant for backward compatibility if present
        mlDashboardCompat: resolve(__dirname, 'js/ml_dashboard.js'),
        modelTraining: resolve(__dirname, 'js/model-training.js'),
        styles: resolve(__dirname, 'css/main.css'),
      }
    }
  }
});

