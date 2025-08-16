const path = require('path');

module.exports = {
  mode: 'production',
  entry: {
    'fa-subset': './static/js/fa-subset.js'
  },
  output: {
    path: path.resolve(__dirname, 'static/dist/js'),
    filename: '[name].bundle.js'
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env']
          }
        }
      }
    ]
  },
  resolve: {
    modules: ['node_modules'],
    extensions: ['.js']
  },
  optimization: {
    minimize: true
  },
  externals: {
    // These will be available globally, so we don't need to bundle them
  }
};
