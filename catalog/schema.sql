CREATE SCHEMA IF NOT EXISTS carddistrict_catalog;
SET search_path TO carddistrict_catalog, public;

CREATE TABLE IF NOT EXISTS categories (
  key text PRIMARY KEY,
  label text NOT NULL,
  kind text NOT NULL CHECK (kind IN ('tcg','sport','non_sport')),
  enabled boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sets (
  id bigserial PRIMARY KEY,
  category_key text NOT NULL REFERENCES categories(key) ON DELETE CASCADE,
  source_key text NOT NULL,
  name text NOT NULL,
  code text NOT NULL DEFAULT '',
  release_date text NOT NULL DEFAULT '',
  card_count integer NOT NULL DEFAULT 0,
  product_type text NOT NULL DEFAULT '',
  region text NOT NULL DEFAULT '',
  brand text NOT NULL DEFAULT '',
  description text NOT NULL DEFAULT '',
  source text NOT NULL DEFAULT '',
  source_url text NOT NULL DEFAULT '',
  image_source_url text NOT NULL DEFAULT '',
  image_asset_id bigint,
  sync_state text NOT NULL DEFAULT 'pending',
  last_synced_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(category_key, source_key)
);

CREATE TABLE IF NOT EXISTS cards (
  id bigserial PRIMARY KEY,
  set_id bigint NOT NULL REFERENCES sets(id) ON DELETE CASCADE,
  category_key text NOT NULL REFERENCES categories(key) ON DELETE CASCADE,
  source_key text NOT NULL,
  name text NOT NULL,
  card_number text NOT NULL DEFAULT '',
  rarity text NOT NULL DEFAULT '',
  variant text NOT NULL DEFAULT '',
  release_date text NOT NULL DEFAULT '',
  market_value numeric(14,2),
  market_currency text NOT NULL DEFAULT '',
  source text NOT NULL DEFAULT '',
  source_url text NOT NULL DEFAULT '',
  image_source_url text NOT NULL DEFAULT '',
  image_asset_id bigint,
  fingerprint text NOT NULL DEFAULT '',
  sync_state text NOT NULL DEFAULT 'pending',
  last_synced_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(set_id, source_key)
);

CREATE TABLE IF NOT EXISTS media_assets (
  id bigserial PRIMARY KEY,
  entity_type text NOT NULL CHECK (entity_type IN ('set','card')),
  entity_id bigint NOT NULL,
  source_url text NOT NULL DEFAULT '',
  storage_provider text NOT NULL DEFAULT '',
  storage_bucket text NOT NULL DEFAULT '',
  storage_path text NOT NULL DEFAULT '',
  public_url text NOT NULL DEFAULT '',
  mime_type text NOT NULL DEFAULT '',
  width integer,
  height integer,
  byte_size bigint,
  sha256 text NOT NULL DEFAULT '',
  perceptual_hash text NOT NULL DEFAULT '',
  status text NOT NULL DEFAULT 'pending',
  last_error text NOT NULL DEFAULT '',
  fetched_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(entity_type, entity_id)
);

CREATE TABLE IF NOT EXISTS sync_jobs (
  id bigserial PRIMARY KEY,
  category_key text NOT NULL REFERENCES categories(key) ON DELETE CASCADE,
  phase text NOT NULL,
  cursor text NOT NULL DEFAULT '',
  status text NOT NULL DEFAULT 'pending',
  processed integer NOT NULL DEFAULT 0,
  failed integer NOT NULL DEFAULT 0,
  last_error text NOT NULL DEFAULT '',
  started_at timestamptz,
  finished_at timestamptz,
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(category_key, phase)
);

CREATE INDEX IF NOT EXISTS sets_category_release_idx ON sets(category_key, release_date DESC);
CREATE INDEX IF NOT EXISTS sets_name_idx ON sets USING gin (to_tsvector('simple', name));
CREATE INDEX IF NOT EXISTS cards_set_idx ON cards(set_id);
CREATE INDEX IF NOT EXISTS cards_category_number_idx ON cards(category_key, card_number);
CREATE INDEX IF NOT EXISTS cards_name_idx ON cards USING gin (to_tsvector('simple', name));
CREATE INDEX IF NOT EXISTS media_status_idx ON media_assets(status, entity_type);

INSERT INTO categories(key,label,kind) VALUES
('pokemon','Pokémon','tcg'),
('one_piece_card_game','One Piece Card Game','tcg'),
('yugioh','Yu-Gi-Oh!','tcg'),
('magic_the_gathering','Magic: The Gathering','tcg'),
('disney_lorcana','Disney Lorcana','tcg'),
('digimon_card_game','Digimon Card Game','tcg'),
('dragon_ball_super','Dragon Ball Super','tcg'),
('flesh_and_blood','Flesh and Blood','tcg'),
('weiss_schwarz','Weiss Schwarz','tcg'),
('union_arena','Union Arena','tcg'),
('star_wars_unlimited','Star Wars Unlimited','tcg'),
('riftbound','Riftbound','tcg'),
('soccer','Fußball','sport'),
('basketball','Basketball','sport'),
('baseball','Baseball','sport'),
('american_football','American Football','sport'),
('ice_hockey','Eishockey','sport'),
('motorsport','Motorsport','sport'),
('tennis','Tennis','sport'),
('golf','Golf','sport'),
('boxing','Boxen','sport'),
('wrestling','Wrestling','sport')
ON CONFLICT (key) DO UPDATE SET label=EXCLUDED.label, kind=EXCLUDED.kind, updated_at=now();
