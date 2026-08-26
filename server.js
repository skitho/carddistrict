import http from 'node:http';
import { createProxyMiddleware } from 'http-proxy-middleware';
import { databaseConfigured, initCatalogSchema, catalogStats } from './catalog/db.js';
import { syncAllCatalog } from './catalog/sync.js';

const PORT = Number(process.env.PORT || 10000);
const TARGET = process.env.UPSTREAM_URL || 'https://cardscope-pro-e4rfnh.v2.appdeploy.ai';
let syncRunning = false;

function applyResponseHeaders(proxyRes, req) {
  proxyRes.headers['x-content-type-options'] = 'nosniff';
  proxyRes.headers['referrer-policy'] = 'strict-origin-when-cross-origin';
  proxyRes.headers['x-frame-options'] = 'SAMEORIGIN';
  delete proxyRes.headers['x-powered-by'];
  const path = String(req.url || '').split('?')[0];
  if (path.startsWith('/api/') || path === '/sw.js') proxyRes.headers['cache-control'] = 'no-store';
  else if (path.startsWith('/assets/') && /\.(?:js|css|woff2?|png|jpe?g|webp|svg)$/i.test(path)) proxyRes.headers['cache-control'] = 'public, max-age=31536000, immutable';
  else if (/\.(?:png|jpe?g|webp|svg|ico|webmanifest)$/i.test(path)) proxyRes.headers['cache-control'] = 'public, max-age=86400, stale-while-revalidate=604800';
  else if (!/\.[a-z0-9]+$/i.test(path) || /\.html?$/i.test(path)) proxyRes.headers['cache-control'] = 'no-cache';
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
      if (location && location.startsWith(TARGET)) proxyRes.headers.location = location.replace(TARGET, `https://${req.headers.host || 'carddistrict.onrender.com'}`);
    },
    error(err, req, res) {
      console.error('CardDistrict upstream error:', err.message);
      if (!res.headersSent) res.writeHead(502, { 'content-type': 'text/plain; charset=utf-8', 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' });
      res.end('CardDistrict wird gerade aktualisiert. Bitte gleich erneut versuchen.');
    }
  }
});

async function json(res, status, body) {
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', 'x-content-type-options': 'nosniff' });
  res.end(JSON.stringify(body));
}

async function kickoffSupabaseCatalog() {
  const base = process.env.SUPABASE_URL || '';
  const bulkToken = process.env.CATALOG_BULK_TOKEN || '';
  const chainToken = process.env.CATALOG_CHAIN_TOKEN || '';
  const pokemonToken = process.env.CATALOG_POKEMON_TOKEN || '';
  if (!base) return;
  const urls = [];
  if (bulkToken) {
    urls.push(`${base}/functions/v1/carddistrict-bulk-provider?token=${encodeURIComponent(bulkToken)}&provider=pokemon_sets`);
    urls.push(`${base}/functions/v1/carddistrict-bulk-provider?token=${encodeURIComponent(bulkToken)}&provider=swu`);
  }
  if (chainToken) {
    for (const category of ['one_piece_card_game','disney_lorcana','magic_the_gathering','yugioh','flesh_and_blood']) {
      urls.push(`${base}/functions/v1/carddistrict-card-chain?token=${encodeURIComponent(chainToken)}&category=${encodeURIComponent(category)}&offset=0&limit=5`);
    }
  }
  if (pokemonToken) urls.push(`${base}/functions/v1/carddistrict-pokemon-chain?token=${encodeURIComponent(pokemonToken)}&offset=0&limit=4`);
  if (!urls.length) return;
  const settled = await Promise.allSettled(urls.map(async u => {
    const r = await fetch(u, { headers: { accept: 'application/json', 'user-agent': 'CardDistrict-Render-Bootstrap/1.0' } });
    const text = await r.text();
    if (!r.ok) throw new Error(`${r.status} ${text.slice(0,180)}`);
    return text.slice(0,240);
  }));
  console.log('Supabase catalog kickoff', settled.map(x => x.status === 'fulfilled' ? x.value : String(x.reason)));
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url || '/', `https://${req.headers.host || 'carddistrict.onrender.com'}`);
  if (url.pathname === '/__carddistrict_health') return json(res, 200, { ok: true, service: 'carddistrict', upstream: TARGET, catalogDatabase: databaseConfigured() });
  if (url.pathname === '/__catalog/status') {
    try { return json(res, 200, await catalogStats()); }
    catch (e) { return json(res, 500, { configured: databaseConfigured(), error: e?.message || String(e) }); }
  }
  if (url.pathname === '/__catalog/sync' && req.method === 'POST') {
    const token = process.env.CATALOG_ADMIN_TOKEN || '';
    if (!token || req.headers.authorization !== `Bearer ${token}`) return json(res, 403, { error: 'forbidden' });
    if (syncRunning) return json(res, 409, { error: 'sync already running' });
    syncRunning = true;
    void syncAllCatalog({ cards: true }).then(x => console.log('catalog sync complete', x?.stats || x)).catch(e => console.error('catalog sync failed', e)).finally(() => { syncRunning = false; });
    return json(res, 202, { accepted: true });
  }
  proxy(req, res);
});

server.keepAliveTimeout = 65000;
server.headersTimeout = 66000;
server.on('upgrade', proxy.upgrade);
server.listen(PORT, '0.0.0.0', async () => {
  console.log(`CardDistrict listening on ${PORT} -> ${TARGET}`);
  if (databaseConfigured()) {
    try { await initCatalogSchema(); console.log('CardDistrict catalog schema ready'); }
    catch (e) { console.error('CardDistrict catalog schema init failed', e); }
  }
  try { await kickoffSupabaseCatalog(); }
  catch (e) { console.error('Supabase catalog kickoff failed', e); }
});
