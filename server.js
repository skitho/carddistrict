import http from 'node:http';
import { createProxyMiddleware } from 'http-proxy-middleware';

const PORT = Number(process.env.PORT || 10000);
const TARGET = String(process.env.UPSTREAM_URL || 'https://cardscope-pro-e4rfnh.v2.appdeploy.ai').replace(/\/$/, '');
const PUBLIC_HOST = 'carddistrict.onrender.com';

function setResponseHeaders(proxyRes, req) {
  const path = String(req.url || '').split('?')[0];
  proxyRes.headers['x-content-type-options'] = 'nosniff';
  proxyRes.headers['referrer-policy'] = 'strict-origin-when-cross-origin';
  proxyRes.headers['x-frame-options'] = 'SAMEORIGIN';
  delete proxyRes.headers['x-powered-by'];
  if (path.startsWith('/api/') || path === '/sw.js') proxyRes.headers['cache-control'] = 'no-store, no-cache, must-revalidate';
  else if (path.startsWith('/assets/') && /\.(?:js|css|woff2?|png|jpe?g|webp|svg)$/i.test(path)) proxyRes.headers['cache-control'] = 'public, max-age=31536000, immutable';
  else proxyRes.headers['cache-control'] = 'no-cache';
}

const proxy = createProxyMiddleware({
  target: TARGET,
  changeOrigin: true,
  secure: true,
  ws: true,
  xfwd: true,
  followRedirects: false,
  on: {
    proxyReq(proxyReq, req) {
      const path = String(req.url || '');
      proxyReq.setHeader('host', new URL(TARGET).host);
      proxyReq.setHeader('x-forwarded-host', req.headers.host || PUBLIC_HOST);
      proxyReq.setHeader('x-forwarded-proto', 'https');
      if (path.startsWith('/api/') || req.method === 'POST' || req.method === 'PUT' || req.method === 'DELETE') {
        proxyReq.setHeader('origin', TARGET);
        proxyReq.setHeader('referer', `${TARGET}/`);
      }
    },
    proxyRes(proxyRes, req) {
      setResponseHeaders(proxyRes, req);
      const location = proxyRes.headers.location;
      if (location && location.startsWith(TARGET)) {
        proxyRes.headers.location = location.replace(TARGET, `https://${req.headers.host || PUBLIC_HOST}`);
      }
      const cookies = proxyRes.headers['set-cookie'];
      if (Array.isArray(cookies)) {
        proxyRes.headers['set-cookie'] = cookies.map(v => v.replace(/;\s*Domain=[^;]+/ig, '').replace(/;\s*SameSite=None/ig, '; SameSite=Lax'));
      }
    },
    error(err, req, res) {
      console.error('CardDistrict proxy error', req.method, req.url, err.message);
      if (!res.headersSent) res.writeHead(502, {'content-type':'application/json; charset=utf-8','cache-control':'no-store'});
      res.end(JSON.stringify({error:'CardDistrict backend temporarily unavailable'}));
    }
  }
});

const server = http.createServer((req, res) => {
  if (req.url === '/__carddistrict_health') {
    res.writeHead(200, {'content-type':'application/json; charset=utf-8','cache-control':'no-store'});
    return res.end(JSON.stringify({ok:true,service:'carddistrict-render-proxy',upstream:TARGET,scannerProxy:true}));
  }
  proxy(req, res);
});

server.keepAliveTimeout = 65000;
server.headersTimeout = 66000;
server.requestTimeout = 120000;
server.on('upgrade', proxy.upgrade);
server.listen(PORT, '0.0.0.0', () => console.log(`CardDistrict proxy listening on ${PORT} -> ${TARGET}`));
