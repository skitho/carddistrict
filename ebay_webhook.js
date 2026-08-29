import http from 'node:http';
import { createHash } from 'node:crypto';

const PORT = Number(process.env.PORT || 10000);
const WEBHOOK_PATH = '/api/ebay/account-deletion';
const VERIFICATION_TOKEN = String(process.env.EBAY_VERIFICATION_TOKEN || '').trim();

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

function tokenValid() {
  return VERIFICATION_TOKEN.length >= 32 && VERIFICATION_TOKEN.length <= 80 && /^[A-Za-z0-9_-]+$/.test(VERIFICATION_TOKEN);
}

async function handleChallenge(req, res, url) {
  const started = Date.now();
  const challenge = url.searchParams.get('challenge_code') || '';
  console.log('EBAY_VALIDATION_REQUEST', {
    method: req.method,
    path: url.pathname,
    challengePresent: Boolean(challenge),
    host: String(req.headers.host || ''),
    tokenConfigured: tokenValid()
  });
  if (!challenge) return sendJson(res, 400, { error: 'challenge_code required' });
  if (!tokenValid()) return sendJson(res, 503, { error: 'verification token not configured' });

  const endpoint = publicEndpoint(req);
  const challengeResponse = createHash('sha256')
    .update(challenge, 'utf8')
    .update(VERIFICATION_TOKEN, 'utf8')
    .update(endpoint, 'utf8')
    .digest('hex');

  console.log('EBAY_VALIDATION_SUCCESS', { endpoint, elapsedMs: Date.now() - started });
  return sendJson(res, 200, { challengeResponse });
}

async function handleNotification(req, res) {
  console.log('EBAY_NOTIFICATION_REQUEST', { method: req.method, path: WEBHOOK_PATH });
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
      console.log('EBAY_NOTIFICATION_ACK', {
        topic: payload?.metadata?.topic || null,
        notificationIdPresent: Boolean(payload?.notification?.notificationId)
      });
    } catch {
      console.warn('EBAY_NOTIFICATION_INVALID_JSON');
    }
  }
  return sendJson(res, 200, { received: true });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url || '/', `http://${req.headers.host || 'localhost'}`);
  if (url.pathname === '/health') return sendJson(res, 200, { ok: true, service: 'cardbrain-ebay-compliance', tokenConfigured: tokenValid(), tokenLength: VERIFICATION_TOKEN.length });
  if (url.pathname === WEBHOOK_PATH && req.method === 'GET') return handleChallenge(req, res, url);
  if (url.pathname === WEBHOOK_PATH && req.method === 'HEAD') {
    res.writeHead(204, { 'cache-control': 'no-store' });
    return res.end();
  }
  if (url.pathname === WEBHOOK_PATH && req.method === 'POST') return handleNotification(req, res);
  if (url.pathname === '/') return sendJson(res, 200, { ok: true, endpoint: WEBHOOK_PATH, tokenConfigured: tokenValid() });
  return sendJson(res, 404, { error: 'not found' });
});

server.keepAliveTimeout = 65000;
server.headersTimeout = 66000;
server.requestTimeout = 30000;
server.listen(PORT, '0.0.0.0', () => {
  console.log('CardBrain eBay compliance listening', { port: PORT, tokenConfigured: tokenValid(), tokenLength: VERIFICATION_TOKEN.length });
});
