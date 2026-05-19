/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // API proxy to backend
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ];
  },

  // Environment variables
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
    NEXT_PUBLIC_APP_NAME: 'PDF Document Extraction',
    NEXT_PUBLIC_APP_VERSION: '1.0.0',
  },

  // Image optimization
  images: {
    domains: ['localhost'],
  },

  // Phase K — Webpack tweaks for the opt-in PDF mode of the Source View.
  // ``pdfjs-dist`` ships its worker as an ES module that Terser cannot
  // minify safely (it uses bare ``import``/``export`` at the worker
  // entry). We exclude that worker file from minification.
  webpack: (config, { dev }) => {
    if (!dev && config.optimization && config.optimization.minimizer) {
      config.optimization.minimizer.forEach((plugin) => {
        if (plugin.constructor && plugin.constructor.name === 'TerserPlugin') {
          const existing = plugin.options.exclude;
          const patterns = Array.isArray(existing)
            ? existing
            : existing
            ? [existing]
            : [];
          patterns.push(/pdf\.worker(\.min)?\.m?js$/);
          plugin.options.exclude = patterns;
        }
      });
    }
    return config;
  },
};

module.exports = nextConfig;
