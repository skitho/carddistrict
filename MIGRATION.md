# CardDistrict Free-Tier migration

## Current production

- Public entry point: https://carddistrict.onrender.com
- Render Web Service: `carddistrict` (Free, Frankfurt)
- Source repository: `skitho/carddistrict`
- Current upstream during migration: existing CardDistrict AppDeploy production

## Free infrastructure prepared

- Existing Render Postgres: `cardscope-pro-db` (Free, Frankfurt)
- Render Key Value cache: `carddistrict-cache` (Free, Frankfurt, allkeys-lru)

## Migration strategy

1. Keep the public Render URL stable while migrating.
2. Move the complete frontend/backend source into this repository.
3. Replace AppDeploy-specific runtime/database/auth APIs with portable Node/Postgres equivalents.
4. Use Postgres for durable card/set/market/news/calendar metadata.
5. Use Key Value only as disposable cache (market/news/calendar hot data).
6. Cache external provider responses and use stale-while-revalidate to minimize latency and rate-limit pressure.
7. Preserve provenance: active listings, guide prices and confirmed sales remain separate data classes.
8. Add health/readiness endpoints and automated smoke tests before removing the upstream proxy.

## Important free-tier constraints

- Free Render Web Services can spin down after inactivity; the public URL can have cold starts.
- Free Key Value is non-persistent and must never be the system of record.
- The current free Postgres instance has an expiry date shown by Render; durable production use will eventually require a persistence decision.
- Image storage should not be placed on the ephemeral web-service filesystem. Use provider-hosted images where permitted and/or a dedicated object-storage/CDN layer.

## Cutover rule

Do not remove the AppDeploy upstream until the standalone implementation passes home, sets, card detail, market, news, calendar, auth and mobile smoke tests on Render.
