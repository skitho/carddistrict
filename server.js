import http from 'node:http';
import { createProxyMiddleware } from 'http-proxy-middleware';

const PORT = Number(process.env.PORT || 10000);
const TARGET = process.env.UPSTREAM_URL || 'https://cardscope-pro-e4rfnh.v2.appdeploy.ai';

const proxy = createProxyMiddleware({
  target: TARGET,
  changeOrigin: true,
  ws: true,
  xfwd: true,
  secure: true,
  followRedirects: false,
  on: {
    proxyReq(proxyReq) {
      proxyReq.setHeader('x-forwarded-host', 'carddistrict.onrender.com');
      proxyReq.setHeader('x-forwarded-proto', 'https');
    },
    proxyRes(proxyRes) {
      const location = proxyRes.headers.location;
      if (location && location.startsWith(TARGET)) {
        proxyRes.headers.location = location.replace(TARGET, 'https://carddistrict.onrender.com');
      }
    },
    error(err, req, res) {
      console.error('CardDistrict upstream error:', err.message);
      if (!res.headersSent) {
        res.writeHead(502, { 'content-type': 'text/plain; charset=utf-8' });
      }
      res.end('CardDistrict wird gerade aktualisiert. Bitte gleich erneut versuchen.');
    }
  }
});

const server = http.createServer((req, res) => {
  if (req.url === '/__carddistrict_health') {
    res.writeHead(200, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' });
    return res.end(JSON.stringify({ ok: true, service: 'carddistrict-render-proxy', upstream: TARGET }));
  }
  proxy(req, res);
});

server.on('upgrade', proxy.upgrade);
server.listen(PORT, '0.0.0.0', () => {
  console.log(`CardDistrict listening on ${PORT} -> ${TARGET}`);
});
