import http from 'node:http';
import { createProxyMiddleware } from 'http-proxy-middleware';

const PORT = Number(process.env.PORT || 10000);
const TARGET = process.env.UPSTREAM_URL || 'https://cardscope-pro-e4rfnh.v2.appdeploy.ai';

function applyResponseHeaders(proxyRes, req) {
  proxyRes.headers['x-content-type-options'] = 'nosniff';
  proxyRes.headers['referrer-policy'] = 'strict-origin-when-cross-origin';
  proxyRes.headers['x-frame-options'] = 'SAMEORIGIN';
  delete proxyRes.headers['x-powered-by'];

  const path = String(req.url || '').split('?')[0];
  if (path.startsWith('/api/') || path === '/sw.js') {
    proxyRes.headers['cache-control'] = 'no-store';
  } else if (path.startsWith('/assets/') && /\.(?:js|css|woff2?|png|jpe?g|webp|svg)$/i.test(path)) {
    proxyRes.headers['cache-control'] = 'public, max-age=31536000, immutable';
  } else if (/\.(?:png|jpe?g|webp|svg|ico|webmanifest)$/i.test(path)) {
    proxyRes.headers['cache-control'] = 'public, max-age=86400, stale-while-revalidate=604800';
  } else if (!/\.[a-z0-9]+$/i.test(path) || /\.html?$/i.test(path)) {
    proxyRes.headers['cache-control'] = 'no-cache';
  }
}

const proxy = createProxyMiddleware({
  target: TARGET,
  changeOrigin: true,
  ws: true,
  xfwd: true,
  secure: true,
  followRedirects: false,
  on: {
    proxyReq(proxyReq, req) {
      proxyReq.setHeader('x-forwarded-host', req.headers.host || 'carddistrict.onrender.com');
      proxyReq.setHeader('x-forwarded-proto', 'https');
    },
    proxyRes(proxyRes, req) {
      applyResponseHeaders(proxyRes, req);
      const location = proxyRes.headers.location;
      if (location && location.startsWith(TARGET)) {
        proxyRes.headers.location = location.replace(TARGET, `https://${req.headers.host || 'carddistrict.onrender.com'}`);
      }
    },
    error(err, req, res) {
      console.error('CardDistrict upstream error:', err.message);
      if (!res.headersSent) {
        res.writeHead(502, {
          'content-type': 'text/plain; charset=utf-8',
          'cache-control': 'no-store',
          'x-content-type-options': 'nosniff'
        });
      }
      res.end('CardDistrict wird gerade aktualisiert. Bitte gleich erneut versuchen.');
    }
  }
});

const server = http.createServer((req, res) => {
  if (req.url === '/__carddistrict_health') {
    res.writeHead(200, {
      'content-type': 'application/json; charset=utf-8',
      'cache-control': 'no-store',
      'x-content-type-options': 'nosniff'
    });
    return res.end(JSON.stringify({ ok: true, service: 'carddistrict-render-proxy', upstream: TARGET }));
  }
  proxy(req, res);
});

server.keepAliveTimeout = 65000;
server.headersTimeout = 66000;
server.on('upgrade', proxy.upgrade);
server.listen(PORT, '0.0.0.0', () => {
  console.log(`CardDistrict listening on ${PORT} -> ${TARGET}`);
});
