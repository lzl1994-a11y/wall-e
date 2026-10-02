const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { test } = require('node:test');
const path = require('node:path');

function page() {
  const context = vm.createContext({
    console, structuredClone, setTimeout, clearTimeout, setInterval, clearInterval,
    document: { addEventListener() {}, querySelector() { return null; } }, window: { setTimeout, setInterval },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/config/app.js'), 'utf8'), context);
  vm.runInContext(`
    renderEyeFields = () => {};
    updateEyePreview = () => {};
    renderEyeDeviceState = () => {};
    setEyeDeviceOnline = () => {};
    setEyeFeedback = () => {};
    setEyeControlsEnabled = () => {};
    showToast = () => {};
    state.eye.values = {...EYE_DEFAULTS};
    state.eye.validValues = {...EYE_DEFAULTS};
  `, context);
  return (code) => vm.runInContext(code, context);
}

test('queued acknowledgement confirms the value encoded at enqueue time', async () => {
  const run = page();
  await run(`(async () => {
    state.eye.values.brightness = 0.2;
    api = async () => ({event:{kind:'ok'}});
    const done = queueEyeCommand('eyeconfig:brightness=0.2', ['brightness']);
    state.eye.values.brightness = 0.8;
    await done;
  })()`);
  assert.equal(run('state.eye.validValues.brightness'), 0.2);
  assert.equal(run('state.eye.values.brightness'), 0.8);
});

test('rejected newer value rolls back to the last device acknowledgement', async () => {
  const run = page();
  await run(`(async () => {
    state.eye.values.brightness = 0.2;
    let acknowledge;
    api = () => new Promise(resolve => acknowledge = resolve);
    const first = queueEyeCommand('eyeconfig:brightness=0.2', ['brightness']);
    await Promise.resolve();
    state.eye.values.brightness = 0.8;
    acknowledge({event:{kind:'ok'}});
    await first;
    api = async () => { throw new Error('EYE:ERR'); };
    await queueEyeCommand('eyeconfig:brightness=0.8', ['brightness']);
  })()`);
  assert.equal(run('state.eye.values.brightness'), 0.2);
});

test('partial reset failure retains accepted fields and queries actual device state', async () => {
  const run = page();
  await run(`(async () => {
    state.eye.values = {...EYE_DEFAULTS, color:'FF0000', ringColor:'00FF00'};
    state.eye.validValues = {...state.eye.values};
    const device = {...state.eye.values};
    let calls = 0;
    api = async (url, options) => {
      if (url.endsWith('/query')) return {state:device, event:{kind:'state'}};
      if (++calls === 2) throw new Error('EYE:ERR');
      device.color = JSON.parse(options.body).command.split('=')[1];
      return {event:{kind:'ok'}};
    };
    resetEyeDefaults();
    await state.eye.commandQueue;
  })()`);
  assert.equal(run('state.eye.values.color'), '00E5FF');
  assert.equal(run('state.eye.validValues.color'), '00E5FF');
  assert.equal(run('state.eye.values.ringColor'), '00FF00');
});

test('device echo is not clamped to narrower UI bounds', () => {
  const run = page();
  const values = run('mergeEyeState({scale:0.25,dots:128,glow:100,breathMs:12000,blinkMs:500})');
  assert.equal(values.scale, 0.25);
  assert.equal(values.dots, 128);
  assert.equal(values.glow, 100);
  assert.equal(values.breathMs, 12000);
  assert.equal(values.blinkMs, 500);
});

test('defaults never send an undocumented automatic-blink setter', () => {
  const run = page();
  assert.equal(run('eyeDefaultCommandList().some(command => command.startsWith("eyeconfig:autoBlink="))'), false);
});

test('automatic blink is enabled by default, stops while speaking, resumes on idle', () => {
  const run = page();
  assert.equal(run('state.eye.previewAutoBlink'), true);
  run(`
    window.setInterval = () => 123;
    restartEyeBlinkTimer();
  `);
  assert.equal(run('state.eye.blinkTimer'), 123);
  run('state.eye.speaking = true; restartEyeBlinkTimer();');
  assert.equal(run('state.eye.blinkTimer'), null);
  run('state.eye.speaking = false; restartEyeBlinkTimer();');
  assert.equal(run('state.eye.blinkTimer'), 123);
  run('state.eye.previewAutoBlink = false; restartEyeBlinkTimer();');
  assert.equal(run('state.eye.blinkTimer'), null);
});

test('numeric command matches its snapshot without rounding fractional device echoes', () => {
  const run = page();
  assert.equal(run('eyeCommandForField("brightness", {...EYE_DEFAULTS, brightness:0.123})'), 'eyeconfig:brightness=0.123');
});

test('empty query cannot mark defaults as device-confirmed state', async () => {
  const run = page();
  run('api = async () => ({event:{kind:"state",fields:{}}});');
  await assert.rejects(run('queryEyeDevice(state.eye.epoch)'), /有效/);
  assert.equal(run('state.eye.loaded'), false);
});

test('unsolicited state updates confirmed values without overwriting an unsent edit', () => {
  const run = page();
  run('state.eye.values.brightness = 0.7; applyEyeState({brightness:0.3, mood:"heart"});');
  assert.equal(run('state.eye.validValues.brightness'), 0.3);
  assert.equal(run('state.eye.values.brightness'), 0.7);
  assert.equal(run('state.eye.values.mood'), 'heart');
});
