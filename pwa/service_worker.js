const CACHE_VERSION = 'bb-v1';
const OFFLINE_URL = '/offline.html';
const STATIC_ASSETS = ['/', '/index.html', OFFLINE_URL, '/manifest.webmanifest',
  '/assets/style.css', '/assets/install.js'];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE_VERSION);
    await cache.addAll(STATIC_ASSETS);
    self.skipWaiting();
  })());
});
self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.map(k => (k !== CACHE_VERSION ? caches.delete(k) : null)));
    self.clients.claim();
  })());
});
self.addEventListener('fetch', (event) => {
  const req = event.request;
  const url = new URL(req.url);
  if (req.method !== 'GET' || url.protocol.startsWith('ws')) return;

  if (req.mode === 'navigate') {
    event.respondWith((async () => {
      try { return await fetch(req); }
      catch {
        const cache = await caches.open(CACHE_VERSION);
        return (await cache.match(OFFLINE_URL)) || new Response('Offline', { status: 503 });
      }
    })());
    return;
  }
  // Let /app/* flow to backend uncached (proxied by Worker)
});
