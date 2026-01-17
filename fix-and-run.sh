#!/bin/bash

echo "🔧 Fixing SPEAKEZ permissions and restarting server..."

# Stop any running Next.js processes
pkill -f "next dev" 2>/dev/null || true
sleep 2

# Clean build cache
echo "Cleaning cache..."
rm -rf .next
rm -rf node_modules/.cache

# Fix permissions on node_modules
echo "Fixing permissions..."
chmod -R u+rwX node_modules 2>/dev/null || true

# Reinstall Next.js if needed
echo "Checking Next.js installation..."
npm list next > /dev/null 2>&1 || npm install next@latest --save

# Start server on port 3001
echo "Starting server on port 3001..."
npm run dev
