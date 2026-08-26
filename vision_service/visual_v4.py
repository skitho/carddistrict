from __future__ import annotations

from typing import Any
import numpy as np
from pydantic import BaseModel, Field
from .visual_v3 import app, ImagePayload, decode, rectify, descriptor, signature, hashes, candidates, download, hsim, orb, ident, now_iso

class ShortlistRequest(BaseModel):
    front: ImagePayload
    category_key: str | None = None
    max_candidates: int = Field(default=6, ge=1, le=12)

class VerifyRequest(BaseModel):
    front: ImagePayload
    candidate: dict[str, Any]

def public_card(row: dict[str, Any], score: float = 0.0) -> dict[str, Any]:
    return {
        'name': str(row.get('player') or row.get('name') or ''),
        'setName': str(row.get('set_name') or ''),
        'cardNumber': str(row.get('card_number') or ''),
        'rarity': 'Rookie / RC' if row.get('rookie') else '',
        'variant': str(row.get('parallel') or row.get('insert_name') or ''),
        'releaseDate': str(row.get('year') or ''),
        'source': 'CardDistrict Visual AI',
        'sourceUrl': str(row.get('source_url') or ''),
        'imageUrl': str(row.get('reference_image_url') or ''),
        'marketValue': 0,
        'marketCurrency': '',
        'visualScore': round(score * 100.0, 1),
        'categoryKey': str(row.get('category_key') or 'unknown'),
        'brand': str(row.get('brand') or ''),
        'team': str(row.get('team') or ''),
        'rookie': bool(row.get('rookie')),
        'insertName': str(row.get('insert_name') or ''),
        'verificationPayload': row,
    }

@app.post('/shortlist')
def shortlist(req: ShortlistRequest):
    scan = rectify(decode(req.front))
    qv = descriptor(scan)
    sig = signature(qv)
    qp, qd = hashes(scan)
    rows = candidates(sig, min(10, max(4, req.max_candidates)), req.category_key)
    out = []
    for row in rows[:req.max_candidates]:
        ss = max(0.0, min(1.0, float(row.get('signature_similarity') or 0.0)))
        out.append(public_card(dict(row), ss))
    if out:
        print('V4 shortlist', {'category': req.category_key or 'all', 'top': [(x['name'], x['categoryKey'], x['visualScore']) for x in out[:5]]}, flush=True)
    return {
        'verified': False,
        'identity': None,
        'card': None,
        'candidates': out,
        'items': [],
        'sales': [],
        'checkedAt': now_iso(),
        'warning': '' if out else 'Noch kein visueller Kandidat im aktiven Referenzindex.',
        'engine': {
            'name': 'CardDistrict Visual AI v4 · shortlist',
            'score': round((float(rows[0].get('signature_similarity') or 0.0) * 100.0), 1) if rows else 0.0,
            'indexedCandidates': len(rows),
            'phash': qp,
            'dhash': qd,
        },
    }

@app.post('/verify')
def verify(req: VerifyRequest):
    row = dict(req.candidate or {})
    url = str(row.get('reference_image_url') or '')
    if not url:
        return {'verified': False, 'identity': None, 'card': None, 'checkedAt': now_iso(), 'warning': 'Referenzbild fehlt.', 'engine': {'name': 'CardDistrict Visual AI v4 · verify', 'score': 0.0}}
    scan = rectify(decode(req.front))
    qv = descriptor(scan)
    qp, qd = hashes(scan)
    try:
        ref = download(url)
    except Exception:
        return {'verified': False, 'identity': None, 'card': public_card(row, 0.0), 'checkedAt': now_iso(), 'warning': 'Referenzbild war vorübergehend nicht erreichbar.', 'engine': {'name': 'CardDistrict Visual AI v4 · verify', 'score': 0.0}}
    rv = descriptor(ref)
    rp, rd = hashes(ref)
    cos = max(0.0, min(1.0, float(np.dot(qv, rv))))
    ps = hsim(qp, str(row.get('phash') or rp))
    ds = hsim(qd, str(row.get('dhash') or rd))
    os = orb(scan, ref)
    ss = max(0.0, min(1.0, float(row.get('signature_similarity') or 0.0)))
    score = .48 * cos + .10 * ss + .15 * ps + .05 * ds + .22 * os
    structural_ok = ps >= .35 or os >= .06
    descriptor_ok = cos >= .70
    shortlist_ok = ss >= .48 or cos >= .86
    verified = bool(score >= .74 and descriptor_ok and structural_ok and shortlist_ok)
    print('V4 verify', {'name': str(row.get('player') or row.get('name') or ''), 'category': str(row.get('category_key') or ''), 'score': round(score, 4), 'cos': round(cos, 4), 'shortlist': round(ss, 4), 'phash': round(ps, 4), 'dhash': round(ds, 4), 'orb': round(os, 4), 'verified': verified}, flush=True)
    return {
        'verified': verified,
        'identity': ident(row, score) if verified else None,
        'card': public_card(row, score),
        'candidates': [],
        'items': [],
        'sales': [],
        'checkedAt': now_iso(),
        'warning': '' if verified else 'Bildähnlichkeit noch nicht eindeutig genug. Nächster Kandidat wird geprüft.',
        'engine': {
            'name': 'CardDistrict Visual AI v4 · verify',
            'score': round(score * 100.0, 1),
            'descriptor': round(cos * 100.0, 1),
            'phash': round(ps * 100.0, 1),
            'dhash': round(ds * 100.0, 1),
            'orb': round(os * 100.0, 1),
        },
    }
