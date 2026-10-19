const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { test } = require('node:test');
const path = require('node:path');

async function page({token = '', authorized = true} = {}) {
  const ids = ['token', 'auth', 'refresh', 'ssid', 'password', 'connect', 'status', 'new-address', 'retry', 'setup', 'connect-form'];
  const nodes = Object.fromEntries(ids.map((id) => [id, {
    value: '', textContent: '', hidden: false, disabled: false, dataset: {}, children: [], listeners: {},
    addEventListener(name, callback) { this.listeners[name] = callback; },
    replaceChildren(...children) { this.children = children; this.value = children[0]?.value || ''; },
    focus() { this.focused = true; },
  }]));
  const calls = [];
  const state = {state: 'setup', ip: '10.42.0.1', detail: '', hotspot_session: true};
  const storage = new Map([['waliConfigToken', token]]);
  const context = vm.createContext({
    console, AbortController, setTimeout, clearTimeout, setInterval() {},
    location: {port: '8080'},
    document: {getElementById: (id) => nodes[id], createElement: () => ({value: '', textContent: ''})},
    sessionStorage: {getItem: (key) => storage.get(key), setItem: (key, value) => storage.set(key, value)},
    fetch: async (url, options) => {
      calls.push({url, options});
      return {ok: authorized, status: authorized ? 200 : 401,
        json: async () => authorized ? {ok: true, ...state, networks: [{ssid: 'shop', signal: 80, security: 'WPA2'}]} : {ok: false, error: 'denied'}};
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/config/wifi.js'), 'utf8'), context);
  await new Promise(setImmediate);
  return {nodes, calls, state, run: (code) => vm.runInContext(code, context)};
}

test('hotspot page automatically loads status and populates real dropdown', async () => {
  const {nodes, calls} = await page();
  assert.deepEqual(calls.map((call) => call.url), ['/api/host-network/status', '/api/host-network/scan']);
  assert.equal(nodes.auth.hidden, true);
  assert.equal(nodes.ssid.children[1].value, 'shop');
  assert.match(nodes.ssid.children[1].textContent, /80%/);
  assert.equal(nodes.refresh.disabled, false);
  assert.equal(calls[0].options.credentials, 'same-origin');
});

test('LAN authentication failure stops scan and focuses token with useful message', async () => {
  const {nodes, calls} = await page({authorized: false});
  assert.equal(calls.length, 1);
  assert.equal(nodes.auth.hidden, false);
  assert.equal(nodes.token.focused, true);
  assert.match(nodes.status.textContent, /配网会话未验证/);
  assert.equal(nodes.refresh.disabled, false);
});

test('existing settings token is reused and connected state keeps polling active', async () => {
  const {nodes, calls, state, run} = await page({token: 'saved-token'});
  assert.equal(calls[0].options.headers['X-Wali-Token'], 'saved-token');
  state.state = 'connected';
  await run('refresh()');
  assert.equal(run('pending'), true);
  assert.equal(nodes.connect.disabled, true);
  state.state = 'ready';
  state.ip = '192.168.0.5';
  state.detail = 'Wi-Fi 配置成功，两端已同步';
  await run('refresh()');
  assert.equal(run('pending'), false);
  assert.equal(nodes.connect.disabled, false);
  assert.match(nodes.status.textContent, /配置成功/);
  assert.equal(nodes['new-address'].href, 'http://192.168.0.5:8080/wifi');
});

test('duplicate submissions issue one request and partial status enables retry', async () => {
  const {nodes, calls, state, run} = await page();
  await run(`Promise.all([submit('connect', {ssid:'shop',password:'password'}, 'connecting'), submit('connect', {ssid:'shop',password:'password'}, 'connecting')])`);
  assert.equal(calls.filter((call) => call.url.endsWith('/connect')).length, 1);
  state.state = 'partial';
  state.detail = '上位机已联网，ESP32 同步失败';
  await run('refresh()');
  assert.equal(nodes.retry.disabled, false);
  assert.match(nodes.status.textContent, /需要重试/);
});
