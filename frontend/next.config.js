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
};

module.exports = nextConfig;
