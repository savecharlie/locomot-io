const CACHE_NAME = 'sediment-v1';
const ASSETS = [
    '/sediment/',
    '/sediment/index.html',
    '/sediment/manifest.json',
    '/sediment/nn_agent.js',
    '/sediment/nn_weights.json',
    '/sediment/music.ogg',
    '/sediment/music.mp3',
    '/sediment/privacy.html',
    '/sediment/icon-48.png',
    '/sediment/icon-72.png',
    '/sediment/icon-96.png',
    '/sediment/icon-144.png',
    '/sediment/icon-192.png',
    '/sediment/icon-512.png',
    '/sediment/icon-maskable-512.png',
    '/sediment/icon-mono-48.png'
];

self.addEventListener('install', (e) => {
    e.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(ASSETS))
            .then(() => self.skipWaiting())
    );
});

self.addEventListener('activate', (e) => {
    e.waitUntil(
        caches.keys().then(keys =>
            Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
        ).then(() => self.clients.claim())
    );
});

self.addEventListener('fetch', (e) => {
    e.respondWith(
        caches.match(e.request).then(cached => cached || fetch(e.request))
    );
});
