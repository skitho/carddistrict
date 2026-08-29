import http from 'node:http';

const PORT = Number(process.env.PORT || 10000);
const HASH_BRIDGE_URL = String(process.env.EBAY_HASH_BRIDGE_URL || 'https://cardbrain-ebay-compliance-06nw89.v2.appdeploy.ai/api/internal/ebay-hash');
const WEBHOOK_PATH = '/api/ebay/account-deletion';

function sendJson(res, status, data) {
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'x-content-type-options': 'nosniff'
  });
  res.end(JSON.stringify(data));
}

function publicEndpoint(req) {
  const forwardedHost = String(req.headers['x-forwarded-host'] || '').split(',')[0].trim();
  const host = forwardedHost || String(req.headers.host || '').trim();
  return `https://${host}${WEBHOOK_PATH}`;
}

async function handleChallenge(req, res, url) {
  const challenge = url.searchParams.get('challenge_code') || '';
  if (!challenge) return sendJson(res, 400, { error: 'challenge_code required' });

  const endpoint = publicEndpoint(req);
  const bridgeUrl = new URL(HASH_BRIDGE_URL);
  bridgeUrl.searchParams.set('challenge_code', challenge);
  bridgeUrl.searchParams.set('endpoint', endpoint);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 9000);
  try {
    const response = await fetch(bridgeUrl, {
      method: 'GET',
      headers: {
        'accept': 'application/json',
        'user-agent': 'CardBrain-eBay-Compliance/1.0'
      },
      signal: controller.signal
    });
    const text = await response.text();
    if (!response.ok) {
      console.error('eBay hash bridge failed', response.status, text.slice(0, 160));
      return sendJson(res, 502, { error: 'challenge bridge failed' });
    }
    let data;
    try { data = JSON.parse(text); } catch { return sendJson(res, 502, { error: 'invalid bridge response' }); }
    if (!data?.challengeResponse || typeof data.challengeResponse !== 'string') return sendJson(res, 502, { error: 'missing challengeResponse' });
    console.log('eBay challenge served', { host: req.headers.host, endpoint });
    return sendJson(res, 200, { challengeResponse: data.challengeResponse });
  } catch (err) {
    console.error('eBay challenge error', err?.message || String(err));
    return sendJson(res, 502, { error: 'challenge service unavailable' });
  } finally {
    clearTimeout(timer);
  }
}

async function handleNotification(req, res) {
  let bytes = 0;
  const chunks = [];
  for await (const chunk of req) {
    bytes += chunk.length;
    if (bytes > 1024 * 1024) return sendJson(res, 413, { error: 'payload too large' });
    chunks.push(chunk);
  }
  const raw = Buffer.concat(chunks).toString('utf8');
  if (raw) {
    try {
      const payload = JSON.parse(raw);
      console.log('eBay deletion notification acknowledged', {
        topic: payload?.metadata?.topic || null,
        notificationIdPresent: Boolean(payload?.notification?.notificationId)
      });
    } catch {
      console.warn('eBay notification body was not JSON');
    }
  }
  return sendJson(res, 200, { received: true });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url || '/', `http://${req.headers.host || 'localhost'}`);
  if (url.pathname === '/health') return sendJson(res, 200, { ok: true, service: 'cardbrain-ebay-compliance' });
  if (url.pathname === WEBHOOK_PATH && req.method === 'GET') return handleChallenge(req, res, url);
  if (url.pathname === WEBHOOK_PATH && req.method === 'POST') return handleNotification(req, res);
  if (url.pathname === '/') return sendJson(res, 200, { ok: true, endpoint: WEBHOOK_PATH });
  return sendJson(res, 404, { error: 'not found' });
});

server.keepAliveTimeout = 65000;
server.headersTimeout = 66000;
server.requestTimeout = 30000;
server.listen(PORT, '0.0.0.0', () => console.log(`CardBrain eBay compliance listening on ${PORT}`));
