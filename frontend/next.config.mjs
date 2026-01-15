/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow images from the backend API
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
        pathname: '/api/v1/files/**',
      },
    ],
  },
};

export default nextConfig;
