import { initCatalogSchema, databaseConfigured, upsertSet, upsertCard, ensureMedia, catalogStats } from './db.js';

const UPSTREAM = process.env.UPSTREAM_URL || 'https://cardscope-pro-e4rfnh.v2.appdeploy.ai';
const CATEGORIES = [
  'pokemon','one_piece_card_game','yugioh','magic_the_gathering','disney_lorcana','digimon_card_game',
  'dragon_ball_super','flesh_and_blood','weiss_schwarz','union_arena','star_wars_unlimited','riftbound',
  'soccer','basketball','baseball','american_football','ice_hockey','motorsport','tennis','golf','boxing','wrestling'
];

const sleep = ms => new Promise(r => setTimeout(r, ms));
async function getJson(path) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), 45000);
  try {
    const r = await fetch(`${UPSTREAM}${path}`, { headers: { accept: 'application/json', 'user-agent': 'CardDistrict-CatalogSync/1.0' }, signal: ctrl.signal });
    if (!r.ok) throw new Error(`${r.status} ${path}`);
    return await r.json();
  } finally { clearTimeout(t); }
}

export async function syncCategory(categoryKey, { cards = true, maxSets = 9999 } = {}) {
  if (!databaseConfigured()) return { categoryKey, configured: false };
  await initCatalogSchema();
  const setPayload = await getJson(`/api/discovery/sets?categoryKey=${encodeURIComponent(categoryKey)}`);
  const sets = Array.isArray(setPayload?.items) ? setPayload.items : [];
  let setCount = 0, cardCount = 0, failures = 0;
  for (const item of sets.slice(0, maxSets)) {
    try {
      const setId = await upsertSet(categoryKey, item);
      await ensureMedia('set', setId, item.imageUrl || '');
      setCount++;
      if (!cards) continue;
      const p = await getJson(`/api/discovery/set-cards?categoryKey=${encodeURIComponent(categoryKey)}&setCode=${encodeURIComponent(item.code || '')}&setName=${encodeURIComponent(item.name || '')}`);
      const list = Array.isArray(p?.items) ? p.items : [];
      for (const card of list) {
        const cardId = await upsertCard(categoryKey, setId, card);
        await ensureMedia('card', cardId, card.imageUrl || '');
        cardCount++;
      }
      await sleep(150);
    } catch (e) {
      failures++;
      console.warn('catalog set sync failed', categoryKey, item?.name, e?.message || e);
    }
  }
  return { categoryKey, sets: setCount, cards: cardCount, failures };
}

export async function syncAllCatalog({ cards = true, maxSetsPerCategory = 9999 } = {}) {
  if (!databaseConfigured()) return { configured: false, reason: 'DATABASE_URL missing' };
  await initCatalogSchema();
  const results = [];
  for (const key of CATEGORIES) {
    try { results.push(await syncCategory(key, { cards, maxSets: maxSetsPerCategory })); }
    catch (e) { results.push({ categoryKey: key, error: e?.message || String(e) }); }
  }
  return { configured: true, results, stats: await catalogStats() };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  syncAllCatalog({ cards: process.env.CATALOG_SYNC_CARDS !== '0' })
    .then(x => { console.log(JSON.stringify(x, null, 2)); process.exit(0); })
    .catch(e => { console.error(e); process.exit(1); });
}
