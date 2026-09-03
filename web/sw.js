const SHELL_CACHE = "clinicalens-shell-v6";
const SHELL_ASSETS = [
  "/",
  "/index.html",
  "/manifest.webmanifest",
  "/config.js",
  "/assets/styles.css?v=20260903-3",
  "/assets/app.js?v=20260903-3",
  "/assets/api.js",
  "/assets/ui.js",
  "/data/multi-organ-pattern.json",
  "/data/metrics.json",
  "/data/test-report.json"
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL_ASSETS)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((key) => key !== SHELL_CACHE).map((key) => caches.delete(key)))));
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin || url.pathname.startsWith("/api/") || event.request.method !== "GET") return;
  event.respondWith(fetch(event.request).then((response) => {
    const copy = response.clone();
    caches.open(SHELL_CACHE).then((cache) => cache.put(event.request, copy));
    return response;
  }).catch(() => caches.match(event.request).then((cached) => cached || caches.match("/index.html"))));
});

self.addEventListener("push", (event) => {
  let payload = { title: "ClinicaLens 提醒", body: "你有一项医生计划相关任务待处理。", url: "/#aftercare" };
  try { payload = { ...payload, ...event.data.json() }; } catch { /* use safe defaults */ }
  event.waitUntil(self.registration.showNotification(payload.title, {
    body: payload.body,
    tag: payload.tag || "clinicalens-reminder",
    data: { url: payload.url || "/#aftercare" },
  }));
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const url = event.notification.data?.url || "/#aftercare";
  event.waitUntil(clients.matchAll({ type: "window", includeUncontrolled: true }).then((windows) => {
    const existing = windows[0];
    return existing ? existing.focus().then(() => existing.navigate(url)) : clients.openWindow(url);
  }));
});
