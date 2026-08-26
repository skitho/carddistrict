import pg from 'pg';
import fs from 'node:fs/promises';
import crypto from 'node:crypto';

const { Pool } = pg;
let pool;

export function databaseConfigured() {
  return Boolean(process.env.DATABASE_URL || process.env.CARDDISTRICT_DATABASE_URL);
}

export function getPool() {
  if (!pool) {
    const connectionString = process.env.CARDDISTRICT_DATABASE_URL || process.env.DATABASE_URL;
    if (!connectionString) throw new Error('CardDistrict catalog database is not configured');
    pool = new Pool({ connectionString, max: 5, idleTimeoutMillis: 30000, connectionTimeoutMillis: 8000 });
  }
  return pool;
}

export async function initCatalogSchema() {
  if (!databaseConfigured()) return false;
  const sql = await fs.readFile(new URL('./schema.sql', import.meta.url), 'utf8');
  await getPool().query(sql);
  return true;
}

export function stableKey(parts) {
  return crypto.createHash('sha1').update(parts.map(v => String(v || '')).join('|')).digest('hex');
}

export async function catalogStats() {
  if (!databaseConfigured()) return { configured: false };
  const db = getPool();
  const q = await db.query(`
    SELECT
      (SELECT count(*)::int FROM carddistrict_catalog.categories) categories,
      (SELECT count(*)::int FROM carddistrict_catalog.sets) sets,
      (SELECT count(*)::int FROM carddistrict_catalog.cards) cards,
      (SELECT count(*)::int FROM carddistrict_catalog.media_assets WHERE status='ready') media_ready,
      (SELECT count(*)::int FROM carddistrict_catalog.media_assets WHERE status<>'ready') media_pending
  `);
  return { configured: true, ...q.rows[0] };
}

export async function upsertSet(categoryKey, item) {
  const db = getPool();
  const sourceKey = item.code || stableKey([categoryKey, item.name, item.releaseDate, item.source]);
  const r = await db.query(`
    INSERT INTO carddistrict_catalog.sets
      (category_key, source_key, name, code, release_date, card_count, product_type, region, brand, description, source, source_url, image_source_url, sync_state, last_synced_at)
    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,'ready',now())
    ON CONFLICT (category_key, source_key) DO UPDATE SET
      name=EXCLUDED.name, code=EXCLUDED.code, release_date=EXCLUDED.release_date,
      card_count=EXCLUDED.card_count, product_type=EXCLUDED.product_type, region=EXCLUDED.region,
      brand=EXCLUDED.brand, description=EXCLUDED.description, source=EXCLUDED.source,
      source_url=EXCLUDED.source_url, image_source_url=EXCLUDED.image_source_url,
      sync_state='ready', last_synced_at=now(), updated_at=now()
    RETURNING id
  `,[categoryKey,sourceKey,item.name||'Unknown set',item.code||'',item.releaseDate||'',Number(item.cardCount)||0,item.productType||'',item.region||'',item.brand||'',item.description||'',item.source||'',item.sourceUrl||'',item.imageUrl||'']);
  return r.rows[0].id;
}

export async function upsertCard(categoryKey, setId, item) {
  const db = getPool();
  const sourceKey = stableKey([item.cardNumber, item.name, item.variant, item.sourceUrl]);
  const r = await db.query(`
    INSERT INTO carddistrict_catalog.cards
      (set_id, category_key, source_key, name, card_number, rarity, variant, release_date, market_value, market_currency, source, source_url, image_source_url, sync_state, last_synced_at)
    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,'ready',now())
    ON CONFLICT (set_id, source_key) DO UPDATE SET
      name=EXCLUDED.name, card_number=EXCLUDED.card_number, rarity=EXCLUDED.rarity,
      variant=EXCLUDED.variant, release_date=EXCLUDED.release_date, market_value=EXCLUDED.market_value,
      market_currency=EXCLUDED.market_currency, source=EXCLUDED.source, source_url=EXCLUDED.source_url,
      image_source_url=EXCLUDED.image_source_url, sync_state='ready', last_synced_at=now(), updated_at=now()
    RETURNING id
  `,[setId,categoryKey,sourceKey,item.name||'Unknown card',item.cardNumber||'',item.rarity||'',item.variant||'',item.releaseDate||'',Number(item.marketValue)||null,item.marketCurrency||'',item.source||'',item.sourceUrl||'',item.imageUrl||'']);
  return r.rows[0].id;
}

export async function ensureMedia(entityType, entityId, sourceUrl) {
  if (!sourceUrl) return;
  await getPool().query(`
    INSERT INTO carddistrict_catalog.media_assets(entity_type, entity_id, source_url, status)
    VALUES ($1,$2,$3,'pending')
    ON CONFLICT (entity_type, entity_id) DO UPDATE SET
      source_url=EXCLUDED.source_url,
      status=CASE WHEN carddistrict_catalog.media_assets.public_url<>'' THEN carddistrict_catalog.media_assets.status ELSE 'pending' END,
      updated_at=now()
  `,[entityType,entityId,sourceUrl]);
}
