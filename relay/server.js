import http from 'node:http';
import crypto from 'node:crypto';
import { WebSocketServer } from 'ws';

const PORT = Number(process.env.PORT || 10000);
const AGENT_TOKEN = String(process.env.CARDBRAIN_AGENT_TOKEN || '').trim();
const MAX_BODY = 40 * 1024 * 1024;
const MAX_QUEUE = 64;
const REQUEST_TIMEOUT_MS = 240000;
const AGENT_PATH = '/__cardbrain_agent';
const HEALTH_PATH = '/__relay/health';

let agent = null;
let inFlight = null;
const queue = [];

function json(res, status, body) {
  const data = Buffer.from(JSON.stringify(body));
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': String(data.length),
    'cache-control': 'no-store',
    'x-content-type-options': 'nosniff',
    'referrer-policy': 'no-referrer',
    'x-frame-options': 'DENY'
  });
  res.end(data);
}

function safeEq(a, b) {
  const aa = Buffer.from(String(a));
  const bb = Buffer.from(String(b));
  return aa.length === bb.length && crypto.timingSafeEqual(aa, bb);
}

function stripHopByHop(headers) {
  const out = {};
  const denied = new Set(['connection','keep-alive','proxy-authenticate','proxy-authorization','te','trailer','transfer-encoding','upgrade','host','content-length']);
  for (const [k, v] of Object.entries(headers || {})) {
    const key = String(k).toLowerCase();
    if (!denied.has(key) && v !== undefined) out[key] = v;
  }
  return out;
}

function clientIp(req) {
  const cf = req.headers['cf-connecting-ip'];
  if (cf) return String(cf).split(',')[0].trim();
  const xff = req.headers['x-forwarded-for'];
  if (xff) return String(xff).split(',')[0].trim();
  return req.socket.remoteAddress || 'unknown';
}

function closePending(message='CardBrain Scanner ist momentan offline.') {
  if (inFlight) {
    clearTimeout(inFlight.timer);
    json(inFlight.res, 503, {ok:false, detail:message});
    inFlight = null;
  }
  while (queue.length) {
    const item = queue.shift();
    json(item.res, 503, {ok:false, detail:message});
  }
}

function pump() {
  if (!agent || agent.readyState !== 1 || inFlight || queue.length === 0) return;
  const item = queue.shift();
  inFlight = item;
  item.timer = setTimeout(() => {
    if (inFlight?.id === item.id) {
      json(item.res, 504, {ok:false, detail:'CardBrain lokale Verarbeitung hat das Zeitlimit überschritten.'});
      inFlight = null;
      pump();
    }
  }, REQUEST_TIMEOUT_MS);
  agent.send(JSON.stringify({
    type:'request',
    id:item.id,
    method:item.method,
    path:item.path,
    headers:item.headers,
    bodyBase64:item.body.toString('base64')
  }));
}

async function readBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY) throw new Error('BODY_TOO_LARGE');
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url || '/', `http://${req.headers.host || 'localhost'}`);
  if (url.pathname === HEALTH_PATH) {
    return json(res, 200, {ok:true, service:'cardbrain-scanner-relay', agentConnected:Boolean(agent && agent.readyState===1), queued:queue.length, busy:Boolean(inFlight)});
  }
  if (url.pathname === AGENT_PATH) return json(res, 426, {ok:false, detail:'WebSocket required'});
  if (!agent || agent.readyState !== 1) {
    return json(res, 503, {ok:false, detail:'CardBrain Scanner ist offline. Der MINIX-PC muss eingeschaltet und CardBrain gestartet sein.'});
  }
  if (queue.length >= MAX_QUEUE) return json(res, 429, {ok:false, detail:'CardBrain Scanner ist ausgelastet. Bitte kurz erneut versuchen.'});
  try {
    const body = await readBody(req);
    const headers = stripHopByHop(req.headers);
    headers['x-forwarded-proto'] = 'https';
    headers['x-forwarded-host'] = String(req.headers.host || '');
    headers['x-forwarded-for'] = clientIp(req);
    headers['x-cardbrain-relay'] = 'render-v1';
    queue.push({
      id:crypto.randomUUID(),
      res,
      method:req.method || 'GET',
      path:url.pathname + url.search,
      headers,
      body,
      timer:null
    });
    pump();
  } catch (err) {
    if (String(err?.message) === 'BODY_TOO_LARGE') return json(res, 413, {ok:false, detail:'Upload zu groß.'});
    return json(res, 500, {ok:false, detail:'Relay request error'});
  }
});

const wss = new WebSocketServer({noServer:true, maxPayload: 60 * 1024 * 1024});

server.on('upgrade', (req, socket, head) => {
  const url = new URL(req.url || '/', `http://${req.headers.host || 'localhost'}`);
  if (url.pathname !== AGENT_PATH || !AGENT_TOKEN) return socket.destroy();
  const provided = String(url.searchParams.get('token') || req.headers['x-cardbrain-agent-token'] || '');
  if (!safeEq(provided, AGENT_TOKEN)) return socket.destroy();
  wss.handleUpgrade(req, socket, head, ws => wss.emit('connection', ws, req));
});

wss.on('connection', ws => {
  if (agent && agent.readyState === 1) agent.close(4001, 'replaced');
  agent = ws;
  console.log('CARDBRAIN_AGENT_CONNECTED');
  ws.on('message', raw => {
    let msg;
    try { msg = JSON.parse(raw.toString()); } catch { return; }
    if (msg?.type === 'ping') {
      try { ws.send(JSON.stringify({type:'pong', ts:Date.now()})); } catch {}
      return;
    }
    if (msg?.type !== 'response' || !inFlight || msg.id !== inFlight.id) return;
    const item = inFlight;
    clearTimeout(item.timer);
    inFlight = null;
    const responseHeaders = stripHopByHop(msg.headers || {});
    for (const [k, v] of Object.entries(responseHeaders)) {
      try { item.res.setHeader(k, v); } catch {}
    }
    item.res.statusCode = Number(msg.status || 502);
    const data = Buffer.from(String(msg.bodyBase64 || ''), 'base64');
    if (!item.res.hasHeader('content-length')) item.res.setHeader('content-length', String(data.length));
    item.res.end(data);
    pump();
  });
  ws.on('close', () => {
    if (agent === ws) {
      agent = null;
      closePending();
      console.log('CARDBRAIN_AGENT_DISCONNECTED');
    }
  });
  ws.on('error', () => {});
  pump();
});

setInterval(() => {
  if (agent?.readyState === 1) {
    try { agent.ping(); } catch {}
  }
}, 20000).unref();

server.listen(PORT, '0.0.0.0', () => {
  console.log(`CardBrain scanner relay listening on ${PORT}`);
  console.log(`Agent token configured: ${AGENT_TOKEN.length >= 32}`);
});
