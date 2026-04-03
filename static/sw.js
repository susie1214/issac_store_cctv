/* Service Worker — 스마트 매장 PWA
   오프라인 시 캐시된 껍데기를 보여주고,
   /video 스트림·/ws WebSocket은 캐시 제외 */

const CACHE = 'smartstore-v1';
const PRECACHE = ['/', '/settings'];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(PRECACHE)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  // 스트림·WebSocket·API는 네트워크만 사용
  if (['/video', '/ws', '/api/'].some(p => url.pathname.startsWith(p))) return;

  e.respondWith(
    fetch(e.request)
      .then(res => {
        if (res.ok) {
          const clone = res.clone();
          caches.open(CACHE).then(c => c.put(e.request, clone));
        }
        return res;
      })
      .catch(() => caches.match(e.request))
  );
});
