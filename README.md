# CardDistrict

Render entry point for CardDistrict.

The current production application remains the upstream while the hosting stack is migrated incrementally. This keeps the public app functional during the transition.

## Render

- Runtime: Node.js
- Build: `npm install`
- Start: `npm start`
- Health: `/__carddistrict_health`
- Preferred service name: `carddistrict`

Set `UPSTREAM_URL` to the current CardDistrict production origin.
