# Fixing Permission Error

If you're seeing "Operation not permitted" errors, try these steps:

## Option 1: Reinstall Dependencies

```bash
rm -rf node_modules package-lock.json
npm install
npm run dev
```

## Option 2: Fix Permissions

```bash
chmod -R u+rw node_modules
npm run dev
```

## Option 3: Use Different Node Version

```bash
nvm use 18  # or another stable version
npm install
npm run dev
```

## Server is Now on Port 3001

The server is configured to run on **port 3001** instead of 3000.

Access it at: **http://localhost:3001**
