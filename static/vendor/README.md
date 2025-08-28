# Vendor assets (self-hosted)

This directory is for self-hosting third-party frontend assets to reduce CDN dependency and enable offline builds.

Suggested structure:

- bootstrap/
  - css/bootstrap.min.css
  - js/bootstrap.bundle.min.js
- chartjs/
  - chart.umd.js
- fontawesome/
  - css/all.min.css
  - webfonts/* (woff2, woff)

How to use with Vite:

1) Place files under the folders above (or similar). Keep versions in filenames if desired, e.g., bootstrap-5.3.3.min.css.
2) Import them in an entry point you control (e.g., static/js/main.js) so they get hashed and included in the manifest:

   import '../vendor/bootstrap/css/bootstrap.min.css';
   import '../vendor/bootstrap/js/bootstrap.bundle.min.js';

   // Chart.js (optional)
   // import '../vendor/chartjs/chart.umd.js';

   // FontAwesome (optional)
   // import '../vendor/fontawesome/css/all.min.css';

3) Rebuild:

   npm run build:ui

4) Reference your page bundles via the Jinja helper asset_url in templates:

   <link rel="stylesheet" href="{{ asset_url('styles.css') }}">
   <script type="module" src="{{ asset_url('app.js') }}"></script>

Notes:
- If you prefer to keep vendor files out of revision control, add them to .gitignore and document a fetch script.
- For FontAwesome, include the "webfonts" directory relative to the CSS file, or adjust the font path via postcss if needed.

