import http from 'node:http';
import { createProxyMiddleware } from 'http-proxy-middleware';
import { databaseConfigured, catalogStats } from './catalog/db.js';

const PORT = Number(process.env.PORT || 10000);
const TARGET = process.env.UPSTREAM_URL || 'https://cardscope-pro-e4rfnh.v2.appdeploy.ai';
const SUPABASE_URL = String(process.env.SUPABASE_URL || '').replace(/\/$/, '');
const SUPABASE_KEY = process.env.SUPABASE_PUBLISHABLE_KEY || process.env.SUPABASE_ANON_KEY || process.env.SUPABASE_KEY || '';
const categoryCache = new Map();

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

function json(res, status, body, maxAge = 0) {
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': maxAge ? `public, max-age=${maxAge}, stale-while-revalidate=${Math.max(60, maxAge * 5)}` : 'no-store',
    'x-content-type-options': 'nosniff',
    'access-control-allow-origin': '*'
  });
  res.end(JSON.stringify(body));
}

function supabaseConfigured() {
  return Boolean(SUPABASE_URL && SUPABASE_KEY);
}

async function sb(table, params = {}) {
  if (!supabaseConfigured()) throw new Error('Supabase catalog is not configured');
  const u = new URL(`${SUPABASE_URL}/rest/v1/${table}`);
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && String(v) !== '') u.searchParams.set(k, String(v));
  const r = await fetch(u, {
    headers: {
      apikey: SUPABASE_KEY,
      authorization: `Bearer ${SUPABASE_KEY}`,
      accept: 'application/json',
      'user-agent': 'CardDistrict-Render-Catalog/2.0'
    }
  });
  const text = await r.text();
  if (!r.ok) throw new Error(`Supabase ${r.status}: ${text.slice(0,240)}`);
  return text ? JSON.parse(text) : [];
}

async function category(key) {
  if (categoryCache.has(key)) return categoryCache.get(key);
  const rows = await sb('cd_categories', { select: 'id,key,label,type', key: `eq.${key}`, active: 'eq.true', limit: 1 });
  const row = rows[0] || null;
  if (row) categoryCache.set(key, row);
  return row;
}

const storedUrl = (bucket, path) => path ? `${SUPABASE_URL}/storage/v1/object/public/${bucket}/${String(path).split('/').map(encodeURIComponent).join('/')}` : '';
const setDto = x => ({
  id: x.id,
  name: x.name || '',
  code: x.code || '',
  releaseDate: x.release_date || x.metadata?.releaseDateRaw || '',
  cardCount: Number(x.card_count) || 0,
  productType: x.product_type || '',
  region: x.region || '',
  source: x.source || 'CardDistrict Catalog',
  sourceUrl: x.source_url || '',
  imageUrl: storedUrl('carddistrict-sets', x.image_storage_path) || x.image_url || '',
  description: x.description || '',
  brand: x.brand || '',
  imageCached: Boolean(x.image_storage_path),
  catalogSource: 'CardDistrict'
});
const cardDto = x => ({
  id: x.id,
  name: x.name || '',
  setName: x.cd_sets?.name || x.set_name || '',
  cardNumber: x.card_number || '',
  rarity: x.rarity || '',
  variant: x.variant || x.parallel || '',
  imageUrl: storedUrl('carddistrict-cards', x.image_storage_path) || x.image_url || '',
  marketValue: Number(x.market_value) || 0,
  marketCurrency: x.market_currency || '',
  source: x.source || 'CardDistrict Catalog',
  sourceUrl: x.source_url || '',
  imageCached: Boolean(x.image_storage_path),
  catalogSource: 'CardDistrict'
});

async function catalogSets(url) {
  const key = url.searchParams.get('categoryKey') || '';
  const cat = await category(key);
  if (!cat) return { status: 400, body: { error: 'Kategorie nicht unterstützt' } };
  const rows = await sb('cd_sets', {
    select: 'id,external_key,code,name,release_date,card_count,product_type,region,brand,description,source,source_url,image_url,image_storage_path,image_status,metadata,last_synced_at',
    category_id: `eq.${cat.id}`,
    order: 'release_date.desc.nullslast,name.asc',
    limit: 500
  });
  return { status: 200, body: { fetchedAt: new Date().toISOString(), items: rows.map(setDto), sourceUrls: [...new Set(rows.map(x => x.source_url).filter(Boolean))], categoryLabel: cat.label, catalog: 'CardDistrict Supabase' } };
}

async function resolveSet(cat, setCode, setName) {
  if (setCode) {
    const a = await sb('cd_sets', { select: 'id,name,code', category_id: `eq.${cat.id}`, code: `eq.${setCode}`, limit: 1 });
    if (a[0]) return a[0];
  }
  if (setName) {
    const a = await sb('cd_sets', { select: 'id,name,code', category_id: `eq.${cat.id}`, name: `eq.${setName}`, limit: 1 });
    if (a[0]) return a[0];
  }
  return null;
}

async function catalogSetCards(url) {
  const key = url.searchParams.get('categoryKey') || '';
  const setCode = (url.searchParams.get('setCode') || '').trim();
  const setName = (url.searchParams.get('setName') || '').trim();
  const cat = await category(key);
  if (!cat) return { status: 400, body: { error: 'Kategorie nicht unterstützt' } };
  const set = await resolveSet(cat, setCode, setName);
  if (!set) return { status: 200, body: { items: [], complete: false, warning: 'Set ist im CardDistrict-Katalog noch nicht vorhanden.', fetchedAt: new Date().toISOString() } };
  const rows = await sb('cd_cards', {
    select: 'id,name,card_number,rarity,variant,parallel,image_url,image_storage_path,image_status,market_value,market_currency,source,source_url',
    set_id: `eq.${set.id}`,
    order: 'card_number.asc,name.asc',
    limit: 1000
  });
  return { status: 200, body: { items: rows.map(x => cardDto({ ...x, set_name: set.name })), complete: rows.length > 0, fetchedAt: new Date().toISOString(), catalog: 'CardDistrict Supabase' } };
}

function safeLike(v) {
  return String(v || '').replace(/[,*()]/g, ' ').trim().slice(0, 100);
}

async function catalogCardSearch(url) {
  const key = url.searchParams.get('categoryKey') || '';
  const q = safeLike(url.searchParams.get('q') || '');
  const cat = await category(key);
  if (!cat) return { status: 400, body: { error: 'Kategorie nicht unterstützt' } };
  if (q.length < 2) return { status: 400, body: { error: 'Bitte mindestens 2 Zeichen suchen' } };
  const rows = await sb('cd_cards', {
    select: 'id,name,card_number,rarity,variant,parallel,image_url,image_storage_path,image_status,market_value,market_currency,source,source_url,set_id,cd_sets(name,release_date)',
    category_id: `eq.${cat.id}`,
    or: `(name.ilike.*${q}*,card_number.ilike.*${q}*)`,
    order: 'name.asc',
    limit: 40
  });
  return { status: 200, body: { items: rows.map(x => ({ ...cardDto(x), releaseDate: x.cd_sets?.release_date || '' })), categoryLabel: cat.label, fetchedAt: new Date().toISOString(), query: q, catalog: 'CardDistrict Supabase' } };
}

async function catalogReleases(url) {
  const key = url.searchParams.get('categoryKey') || '';
  const cat = await category(key);
  if (!cat) return { status: 400, body: { error: 'Kategorie nicht unterstützt' } };
  const rows = await sb('cd_sets', {
    select: 'id,name,release_date,product_type,region,source,source_url,image_url,image_storage_path',
    category_id: `eq.${cat.id}`,
    release_date: 'not.is.null',
    order: 'release_date.desc',
    limit: 250
  });
  const items = rows.map(x => ({ name: x.name, releaseDate: x.release_date || '', productType: x.product_type || '', region: x.region || '', source: x.source || 'CardDistrict Catalog', sourceUrl: x.source_url || '', imageUrl: storedUrl('carddistrict-sets', x.image_storage_path) || x.image_url || '' }));
  return { status: 200, body: { fetchedAt: new Date().toISOString(), items, sourceUrls: [...new Set(rows.map(x => x.source_url).filter(Boolean))], catalog: 'CardDistrict Supabase' } };
}

async function ownCatalogRoute(url) {
  if (url.pathname === '/api/discovery/sets') return catalogSets(url);
  if (url.pathname === '/api/discovery/set-cards') return catalogSetCards(url);
  if (url.pathname === '/api/discovery/cards') return catalogCardSearch(url);
  if (url.pathname === '/api/discovery/releases') return catalogReleases(url);
  return null;
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url || '/', `https://${req.headers.host || 'carddistrict.onrender.com'}`);
  if (url.pathname === '/__carddistrict_health') return json(res, 200, { ok: true, service: 'carddistrict', upstream: TARGET, supabaseCatalog: supabaseConfigured(), legacyCatalogDatabase: databaseConfigured() });
  if (url.pathname === '/__catalog/status') {
    try {
      const stats = await catalogStats().catch(() => ({}));
      return json(res, 200, { ...stats, supabaseConfigured: supabaseConfigured(), importPaused: true, primaryCatalog: 'supabase' });
    } catch (e) { return json(res, 500, { configured: databaseConfigured(), supabaseConfigured: supabaseConfigured(), error: e?.message || String(e) }); }
  }
  if (url.pathname === '/__catalog/sync' && req.method === 'POST') return json(res, 423, { error: 'catalog imports are paused by free-tier storage protection' });
  if (req.method === 'GET' && supabaseConfigured()) {
    try {
      const handled = await ownCatalogRoute(url);
      if (handled) return json(res, handled.status, handled.body, handled.status === 200 ? 300 : 0);
    } catch (e) {
      console.error('Supabase catalog route failed', url.pathname, e);
      // Fall through to the proven upstream for availability while migration continues.
    }
  }
  proxy(req, res);
});

server.keepAliveTimeout = 65000;
server.headersTimeout = 66000;
server.on('upgrade', proxy.upgrade);
server.listen(PORT, '0.0.0.0', () => console.log(`CardDistrict listening on ${PORT} -> ${TARGET}; primary catalog: ${supabaseConfigured() ? 'Supabase' : 'upstream'}`));
