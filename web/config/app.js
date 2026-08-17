"use strict";

const DEFAULT_ACCESS_TOKEN = "123456";

const state = {
  config: null,
  secretFields: {},
  dirtyModules: new Set(),
  loading: false,
  usbDevices: [],
  usbLoading: false,
  cameraPreview: {
    active: false,
    timer: null,
    objectUrl: null,
    requesting: false,
    frameCount: 0,
  },
};

const CAMERA_PREVIEW_POLL_MS = 180;

const MODULE_ROOTS = Object.freeze({
  runtime: "launch",
  pipeline: "pipeline",
  asr: "asr",
  wake_word: "wake_word",
  vad: "vad",
  tts: "tts",
  llm: "llm",
  system_prompt: "system_prompt",
  hardware: "hardware",
  serial: "serial",
  i2c: "i2c",
  remote_control: "remote_control",
  vision: "vision",
  servos: "servos",
  motors: "motors",
  usb_devices: "usb_devices",
});

const MODULE_LABELS = Object.freeze({
  runtime: "运行",
  pipeline: "对话链路",
  asr: "ASR",
  wake_word: "唤醒词",
  vad: "VAD",
  tts: "TTS",
  llm: "LLM",
  system_prompt: "系统提示词",
  hardware: "运动硬件后端",
  serial: "串口",
  i2c: "I²C",
  remote_control: "手柄遥控",
  vision: "视觉",
  servos: "舵机",
  motors: "电机",
  usb_devices: "USB 设备",
});

const ASR_DEFAULTS = Object.freeze({
  zhipu: {
    model: "",
    url: "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions",
    api_key: "",
  },
  aliyun: { model: "", api_key: "" },
  baidu: {
    app_id: "",
    api_key: "",
    dev_pid: 15372,
    cuid: "wali-x3",
    url: "wss://vop.baidu.com/realtime_asr",
  },
});

const LOCAL_ASR_DEFAULTS = Object.freeze({
  sherpa_onnx_zipformer: {
    encoder: "",
    decoder: "",
    joiner: "",
    tokens: "",
    num_threads: 2,
  },
  sherpa_onnx_paraformer: {
    model: "",
    tokens: "",
    num_threads: 2,
  },
  sherpa_onnx_sensevoice: {
    model: "",
    tokens: "",
    language: "auto",
    use_itn: true,
    num_threads: 2,
  },
  sherpa_onnx_whisper: {
    encoder: "",
    decoder: "",
    tokens: "",
    language: "zh",
    num_threads: 2,
  },
  faster_whisper: {
    model_path: "",
    language: "zh",
    device: "cpu",
    compute_type: "int8",
  },
});

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function getToken() {
  return $("#access-token").value.trim();
}

function apiHeaders(includeJson = false) {
  const headers = {};
  const token = getToken();
  if (token) headers["X-Wali-Token"] = token;
  if (includeJson) headers["Content-Type"] = "application/json";
  return headers;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    ...options,
    headers: { ...apiHeaders(Boolean(options.body)), ...(options.headers || {}) },
  });
  let payload = {};
  try { payload = await response.json(); } catch (_) { /* empty or non-JSON response */ }
  if (!response.ok) {
    const error = new Error(payload.error || `请求失败 (${response.status})`);
    error.details = payload.details || [];
    error.status = response.status;
    throw error;
  }
  return payload;
}

function setCameraPreviewControls(active) {
  $("#camera-preview-start").disabled = active;
  $("#camera-preview-stop").disabled = !active;
}

function clearCameraPreviewImage() {
  const image = $("#camera-preview-image");
  image.hidden = true;
  image.removeAttribute("src");
  if (state.cameraPreview.objectUrl) {
    URL.revokeObjectURL(state.cameraPreview.objectUrl);
    state.cameraPreview.objectUrl = null;
  }
}

function renderCameraPreviewStatus(status = {}) {
  const previewState = status.state || "stopped";
  const startingText = {
    resolving: "正在查找摄像头",
    launching: "正在启动采集进程",
    opening: status.device ? `正在打开 ${status.device}` : "正在打开摄像头",
    waiting_frame: status.device ? `等待 ${status.device} 首帧` : "等待摄像头首帧",
  }[status.phase] || "正在连接";
  const statusText = {
    starting: startingText,
    running: "实时预览中",
    stopping: "正在停止",
    stopped: "已停止",
    error: status.error || "摄像头预览失败",
  }[previewState] || previewState;
  const dot = $("#camera-preview-dot");
  dot.className = `preview-status-dot ${previewState}`;
  $("#camera-preview-status").textContent = statusText;
  $("#camera-preview-status").title = statusText;
  $("#camera-preview-device").textContent = status.device || "—";
  $("#camera-preview-resolution").textContent = status.width && status.height
    ? `${status.width} × ${status.height}`
    : "—";
  $("#camera-preview-fps").textContent = status.fps ? `${status.fps} FPS` : "—";
}

function scheduleCameraPreviewPoll(delay = CAMERA_PREVIEW_POLL_MS) {
  clearTimeout(state.cameraPreview.timer);
  if (!state.cameraPreview.active) return;
  state.cameraPreview.timer = window.setTimeout(pollCameraPreviewFrame, delay);
}

async function refreshCameraPreviewStatus() {
  try {
    const status = await api("/api/camera-preview/status");
    renderCameraPreviewStatus(status);
    if (["error", "stopped"].includes(status.state)) {
      state.cameraPreview.active = false;
      setCameraPreviewControls(false);
    }
  } catch (_) {
    // Frame polling owns connection error reporting.
  }
}

async function pollCameraPreviewFrame() {
  if (!state.cameraPreview.active || state.cameraPreview.requesting) return;
  state.cameraPreview.requesting = true;
  try {
    const response = await fetch(`/api/camera-preview/frame?t=${Date.now()}`, {
      cache: "no-store",
      headers: apiHeaders(false),
    });
    if (!response.ok) {
      let payload = {};
      try { payload = await response.json(); } catch (_) { /* non-JSON error */ }
      if (response.status === 503 && ["starting", "running"].includes(payload.state)) {
        renderCameraPreviewStatus(payload);
        return;
      }
      if (response.status === 503 && payload.state === "stopped") {
        state.cameraPreview.active = false;
        setCameraPreviewControls(false);
        renderCameraPreviewStatus(payload);
        clearCameraPreviewImage();
        $("#camera-preview-placeholder").textContent = payload.error || "预览已停止";
        $("#camera-preview-placeholder").hidden = false;
        return;
      }
      const error = new Error(payload.error || `摄像头画面请求失败 (${response.status})`);
      error.status = response.status;
      throw error;
    }

    const blob = await response.blob();
    const nextUrl = URL.createObjectURL(blob);
    const previousUrl = state.cameraPreview.objectUrl;
    const image = $("#camera-preview-image");
    image.onload = () => {
      if (previousUrl) URL.revokeObjectURL(previousUrl);
    };
    image.src = nextUrl;
    image.hidden = false;
    $("#camera-preview-placeholder").hidden = true;
    state.cameraPreview.objectUrl = nextUrl;
    state.cameraPreview.frameCount += 1;
    if (state.cameraPreview.frameCount % 8 === 1) refreshCameraPreviewStatus();
  } catch (error) {
    state.cameraPreview.active = false;
    setCameraPreviewControls(false);
    clearCameraPreviewImage();
    renderCameraPreviewStatus({ state: "error", error: error.message });
    $("#camera-preview-placeholder").textContent = error.message;
    $("#camera-preview-placeholder").hidden = false;
    showToast(error.message, "error");
  } finally {
    state.cameraPreview.requesting = false;
    scheduleCameraPreviewPoll();
  }
}

async function startCameraPreview() {
  if (state.cameraPreview.active) return;
  state.cameraPreview.active = true;
  state.cameraPreview.frameCount = 0;
  setCameraPreviewControls(true);
  clearCameraPreviewImage();
  renderCameraPreviewStatus({ state: "starting" });
  $("#camera-preview-placeholder").textContent = "正在连接摄像头";
  $("#camera-preview-placeholder").hidden = false;
  try {
    const status = await api("/api/camera-preview/start", {
      method: "POST",
      body: JSON.stringify({}),
    });
    renderCameraPreviewStatus(status);
    scheduleCameraPreviewPoll(0);
  } catch (error) {
    state.cameraPreview.active = false;
    setCameraPreviewControls(false);
    clearCameraPreviewImage();
    renderCameraPreviewStatus({ state: "error", error: error.message });
    $("#camera-preview-placeholder").textContent = error.message;
    showToast(error.message, "error");
  }
}

async function stopCameraPreview({ quiet = false } = {}) {
  state.cameraPreview.active = false;
  clearTimeout(state.cameraPreview.timer);
  setCameraPreviewControls(false);
  renderCameraPreviewStatus({ state: "stopping" });
  try {
    const status = await api("/api/camera-preview/stop", {
      method: "POST",
      body: JSON.stringify({}),
    });
    renderCameraPreviewStatus(status);
  } catch (error) {
    renderCameraPreviewStatus({ state: "error", error: error.message });
    if (!quiet) showToast(error.message, "error");
  }
}

async function reconnectCameraPreview() {
  await stopCameraPreview({ quiet: true });
  await startCameraPreview();
}

function stopCameraPreviewOnPageExit() {
  if (!state.cameraPreview.active) return;
  state.cameraPreview.active = false;
  fetch("/api/camera-preview/stop", {
    method: "POST",
    headers: apiHeaders(true),
    body: JSON.stringify({}),
    keepalive: true,
  }).catch(() => {});
}

async function changeAccessToken() {
  const input = $("#new-access-token");
  const newToken = input.value.trim();
  if (!newToken) {
    showToast("请输入新的访问令牌", "error");
    input.focus();
    return;
  }
  const button = $("#change-token-button");
  button.disabled = true;
  try {
    const payload = await api("/api/access-token", {
      method: "POST",
      body: JSON.stringify({ new_token: newToken }),
    });
    $("#access-token").value = newToken;
    sessionStorage.setItem("waliConfigToken", newToken);
    input.value = "";
    showToast(payload.message || "访问令牌已修改");
    setConnection(true, "访问令牌已更新");
  } catch (error) {
    showErrors(error);
    showToast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

function getPath(target, path) {
  return path.split(".").reduce((value, key) => value == null ? undefined : value[key], target);
}

function setPath(target, path, value) {
  const keys = path.split(".");
  let cursor = target;
  keys.slice(0, -1).forEach((key) => {
    if (!cursor[key] || typeof cursor[key] !== "object") cursor[key] = {};
    cursor = cursor[key];
  });
  cursor[keys.at(-1)] = value;
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function ensureRemoteControlConfig() {
  const defaults = { servo_step_size: 40.0, update_rate_hz: 20 };
  if (!state.config.remote_control || typeof state.config.remote_control !== "object") {
    state.config.remote_control = {};
  }
  Object.entries(defaults).forEach(([key, value]) => {
    if (state.config.remote_control[key] === undefined) state.config.remote_control[key] = value;
  });
}

function ensureHardwareConfig() {
  if (!state.config.hardware || typeof state.config.hardware !== "object") {
    state.config.hardware = {};
  }
  if (!["serial_mcu", "ubuntu_i2c"].includes(state.config.hardware.backend)) {
    state.config.hardware.backend = "serial_mcu";
  }
}

function ensureVadConfig() {
  const defaults = {
    provider: "webrtc",
    aggressiveness: 3,
    model_path: "models/silero_vad.onnx",
    threshold: 0.5,
    silence_sec: 0.5,
  };
  if (!state.config.vad || typeof state.config.vad !== "object") {
    state.config.vad = {};
  }
  Object.entries(defaults).forEach(([key, value]) => {
    if (state.config.vad[key] === undefined) state.config.vad[key] = value;
  });
  if (!["webrtc", "silero"].includes(state.config.vad.provider)) {
    state.config.vad.provider = "webrtc";
  }
}

function ensureLlmConfig() {
  if (!state.config.llm || typeof state.config.llm !== "object") {
    state.config.llm = {};
  }
  if (!["fast", "default"].includes(state.config.llm.reasoning_effort)) {
    state.config.llm.reasoning_effort = "fast";
  }
}

function ensureUsbDeviceConfig() {
  if (!state.config.usb_devices || typeof state.config.usb_devices !== "object") {
    state.config.usb_devices = {};
  }
}

function selectorKey(selector) {
  if (!selector || typeof selector !== "object") return "";
  return JSON.stringify({
    vendor_id: String(selector.vendor_id || "").toLowerCase(),
    product_id: String(selector.product_id || "").toLowerCase(),
    ...(selector.serial_number ? { serial_number: selector.serial_number } : {}),
    ...(!selector.serial_number && selector.port_path ? { port_path: selector.port_path } : {}),
  });
}

function usbInterfaceSummary(device) {
  const interfaces = device.interfaces || {};
  const parts = [];
  if (interfaces.video?.length) parts.push(interfaces.video.join(", "));
  if (interfaces.serial?.length) parts.push(interfaces.serial.join(", "));
  if (interfaces.audio_cards?.length) parts.push(`audio card ${interfaces.audio_cards.join(", ")}`);
  return parts.join(" | ") || "USB 设备已连接，未发现可用接口";
}

function renderUsbSelectors() {
  if (!state.config) return;
  $$('[data-usb-role]').forEach((select) => {
    const role = select.dataset.usbRole;
    const saved = state.config.usb_devices?.[role] || null;
    const savedKey = selectorKey(saved);
    const options = [];
    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = "使用代码默认识别";
    options.push(defaultOption);

    state.usbDevices.forEach((device) => {
      const option = document.createElement("option");
      option.value = selectorKey(device.selector);
      option.textContent = `${device.label} - ${usbInterfaceSummary(device)}`;
      options.push(option);
    });

    if (savedKey && !options.some((option) => option.value === savedKey)) {
      const offlineOption = document.createElement("option");
      offlineOption.value = savedKey;
      offlineOption.textContent = `${saved.vendor_id}:${saved.product_id}（已保存，当前离线）`;
      options.push(offlineOption);
    }
    select.replaceChildren(...options);
    select.value = savedKey;

    const selectedDevice = state.usbDevices.find((device) => selectorKey(device.selector) === savedKey);
    const detail = $(`[data-usb-detail="${role}"]`);
    if (detail) {
      detail.textContent = selectedDevice
        ? usbInterfaceSummary(selectedDevice)
        : savedKey ? "设备当前离线，插回后会自动恢复" : "未绑定，使用代码默认识别逻辑";
    }
  });
}

async function loadUsbDevices({ quiet = false } = {}) {
  if (!state.config || state.usbLoading) return;
  state.usbLoading = true;
  const button = $("#refresh-usb-devices");
  if (button) button.disabled = true;
  $("#usb-scan-status").textContent = "正在扫描 USB 设备...";
  try {
    const payload = await api("/api/usb-devices");
    state.usbDevices = payload.devices || [];
    renderUsbSelectors();
    $("#usb-scan-status").textContent = `发现 ${state.usbDevices.length} 个物理 USB 设备，更新于 ${new Date().toLocaleTimeString()}`;
    if (!quiet) showToast(`已发现 ${state.usbDevices.length} 个 USB 设备`);
  } catch (error) {
    $("#usb-scan-status").textContent = "USB 扫描失败";
    if (!quiet) {
      showErrors(error);
      showToast(error.message, "error");
    }
  } finally {
    state.usbLoading = false;
    if (button) button.disabled = false;
  }
}

function ensureAsrConfigs() {
  const asr = state.config.asr || (state.config.asr = {});
  const configuredMode = asr.mode ?? asr.type;
  asr.mode = ["cloud", "local"].includes(configuredMode) ? configuredMode : "cloud";
  const provider = Object.hasOwn(ASR_DEFAULTS, asr.provider) ? asr.provider : "zhipu";
  asr.provider = provider;
  const engine = Object.hasOwn(LOCAL_ASR_DEFAULTS, asr.engine)
    ? asr.engine
    : "sherpa_onnx_zipformer";
  asr.engine = engine;

  const hadProviderConfig = asr[provider] && typeof asr[provider] === "object";

  Object.entries(ASR_DEFAULTS).forEach(([name, defaults]) => {
    if (!asr[name] || typeof asr[name] !== "object") asr[name] = {};
    if (name === provider && !hadProviderConfig) {
      if (asr.model) asr[name].model = asr.model;
      if (name === "zhipu" && asr.url) asr[name].url = asr.url;
    }
    Object.entries(defaults).forEach(([key, value]) => {
      if (asr[name][key] === undefined) asr[name][key] = value;
    });
  });

  Object.entries(LOCAL_ASR_DEFAULTS).forEach(([name, defaults]) => {
    if (!asr[name] || typeof asr[name] !== "object") asr[name] = {};
    Object.entries(defaults).forEach(([key, value]) => {
      if (asr[name][key] === undefined) asr[name][key] = value;
    });
  });

  // A legacy flat secret belongs to the provider that was active in that file.
  if (state.secretFields["asr.key"] && state.secretFields[`asr.${provider}.api_key`] === undefined) {
    state.secretFields[`asr.${provider}.api_key`] = true;
  }
}

function updateAsrModePanels(mode = $("#asr-mode")?.value) {
  $$('[data-asr-mode-panel]').forEach((panel) => {
    panel.hidden = panel.dataset.asrModePanel !== mode;
  });
}

function updateAsrProviderPanels(provider = $("#asr-provider")?.value) {
  $$('[data-asr-provider-panel]').forEach((panel) => {
    panel.hidden = panel.dataset.asrProviderPanel !== provider;
  });
}

function updateLocalAsrEnginePanels(engine = $("#asr-engine")?.value) {
  $$('[data-local-asr-engine-panel]').forEach((panel) => {
    panel.hidden = panel.dataset.localAsrEnginePanel !== engine;
  });
}

function updateVadProviderPanels(provider = $("#vad-provider")?.value) {
  $$('[data-vad-provider-panel]').forEach((panel) => {
    panel.hidden = panel.dataset.vadProviderPanel !== provider;
  });
}

function updateHardwareBackendPanels(backend = $("#hardware-backend")?.value) {
  $$('[data-hardware-backend-panel]').forEach((panel) => {
    panel.hidden = panel.dataset.hardwareBackendPanel !== backend;
  });
}

function moduleContainer(module) {
  return $(`[data-module="${module}"]`);
}

function prepareModuleFeedbacks() {
  $$('[data-save-module]').forEach((button) => {
    const status = document.createElement("span");
    status.className = "module-feedback";
    status.dataset.feedbackModule = button.dataset.saveModule;
    status.setAttribute("aria-live", "polite");
    button.before(status);
  });
}

function setModuleFeedback(module, message = "", kind = "") {
  const status = $(`[data-feedback-module="${module}"]`);
  if (!status) return;
  status.textContent = message;
  status.className = `module-feedback${kind ? ` ${kind}` : ""}`;
}

function clearModuleFeedbacks() {
  $$('[data-feedback-module]').forEach((status) => {
    status.textContent = "";
    status.className = "module-feedback";
  });
}

function setConfigControlsEnabled(enabled) {
  $$('[data-module] [data-path]').forEach((input) => { input.disabled = !enabled; });
  for (const id of ["#add-servo", "#add-motor"]) {
    const button = $(id);
    if (button) button.disabled = !enabled;
  }
  $$('[data-usb-role]').forEach((select) => { select.disabled = !enabled; });
  if ($("#refresh-usb-devices")) $("#refresh-usb-devices").disabled = !enabled;
}

function clearConfigurationView() {
  state.config = null;
  state.secretFields = {};
  $$('[data-path]').forEach((input) => {
    if (input.type === "checkbox") input.checked = false;
    else input.value = "";
  });
  $$('[data-secret-status]').forEach((status) => { status.textContent = "配置尚未读取"; });
  $("#servo-table").replaceChildren();
  $("#motor-table").replaceChildren();
  $("#servo-count").textContent = "0 个舵机";
  $("#motor-count").textContent = "0 个电机";
  $("#config-path").textContent = "—";
  $("#config-path").title = "";
  $("#modified-at").textContent = "—";
  updateAsrProviderPanels("");
  updateVadProviderPanels("");
  updateHardwareBackendPanels("");
  state.usbDevices = [];
  renderUsbSelectors();
  clearAllDirty();
  clearModuleFeedbacks();
}

function updateDirtyIndicator() {
  const count = state.dirtyModules.size;
  const indicator = $("#dirty-indicator");
  indicator.textContent = count ? `有未保存修改（${count} 个模块）` : "未修改";
  indicator.classList.toggle("dirty", count > 0);
  $$('[data-save-module]').forEach((button) => {
    button.classList.toggle("dirty", state.dirtyModules.has(button.dataset.saveModule));
  });
}

function markDirty(module) {
  if (!state.config || state.loading || !MODULE_ROOTS[module]) return;
  state.dirtyModules.add(module);
  setModuleFeedback(module, "有未保存修改", "dirty");
  updateDirtyIndicator();
}

function clearDirty(module) {
  state.dirtyModules.delete(module);
  updateDirtyIndicator();
}

function clearAllDirty() {
  state.dirtyModules.clear();
  updateDirtyIndicator();
}

function setConnection(online, text) {
  $("#connection-text").textContent = text;
  $("#connection-dot").className = `status-dot ${online ? "online" : "offline"}`;
}

function showToast(message, kind = "success") {
  const toast = $("#toast");
  toast.textContent = message;
  toast.className = `toast ${kind} show`;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { toast.className = "toast"; }, 3200);
}

function showErrors(error) {
  const details = error.details?.length ? error.details : [error.message];
  const list = $("#error-list");
  list.replaceChildren(...details.map((detail) => {
    const li = document.createElement("li");
    li.textContent = detail;
    return li;
  }));
  $("#error-box").hidden = false;
}

function inputValue(input) {
  if (input.type === "checkbox") return input.checked;
  const type = input.dataset.type;
  if (type === "integer") return Number.parseInt(input.value, 10);
  if (type === "number") return Number.parseFloat(input.value);
  return input.value;
}

function populateFields(root = document) {
  $$('[data-path]', root).forEach((input) => {
    const value = getPath(state.config, input.dataset.path);
    if (input.dataset.secret === "true") {
      input.value = "";
      return;
    }
    if (input.type === "checkbox") input.checked = Boolean(value);
    else input.value = value ?? "";
  });
  $$('[data-secret-status]', root).forEach((node) => {
    const configured = Boolean(state.secretFields[node.dataset.secretStatus]);
    node.textContent = configured ? "当前已配置；输入新值可覆盖" : "当前未配置";
  });
}

function createTableInput(value, type, onChange) {
  const input = document.createElement("input");
  input.type = type === "boolean" ? "checkbox" : (type === "string" ? "text" : "number");
  if (type === "boolean") input.checked = Boolean(value);
  else input.value = value ?? "";
  if (type === "integer") input.step = "1";
  input.addEventListener("input", () => {
    const next = type === "boolean" ? input.checked : type === "integer" ? Number.parseInt(input.value, 10) : input.value;
    onChange(next);
  });
  return input;
}

function appendCell(row, child, className = "") {
  const cell = document.createElement("td");
  if (className) cell.className = className;
  cell.append(child);
  row.append(cell);
}

function removeButton(onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "remove-row";
  button.textContent = "×";
  button.title = "删除";
  button.addEventListener("click", onClick);
  return button;
}

function renderServos() {
  const body = $("#servo-table");
  body.replaceChildren();
  const servos = state.config.servos || [];
  servos.forEach((servo, index) => {
    const row = document.createElement("tr");
    [
      ["id", "integer"], ["name", "string"], ["limit_1", "integer"],
      ["limit_2", "integer"], ["init", "integer"],
    ].forEach(([key, type]) => appendCell(row, createTableInput(servo[key], type, (value) => {
      servo[key] = value;
      markDirty("servos");
    })));
    appendCell(row, removeButton(() => {
      servos.splice(index, 1);
      renderServos();
      markDirty("servos");
    }));
    body.append(row);
  });
  $("#servo-count").textContent = `${servos.length} 个舵机`;
}

function renderMotors() {
  const body = $("#motor-table");
  body.replaceChildren();
  const motors = state.config.motors || [];
  motors.forEach((motor, index) => {
    const row = document.createElement("tr");
    [["id", "integer"], ["name", "string"], ["max_speed", "integer"], ["neutral_speed", "integer"]]
      .forEach(([key, type]) => appendCell(row, createTableInput(motor[key], type, (value) => {
        motor[key] = value;
        markDirty("motors");
      })));
    appendCell(row, createTableInput(motor.invert_direction, "boolean", (value) => {
      motor.invert_direction = value;
      markDirty("motors");
    }), "table-check");
    appendCell(row, removeButton(() => {
      motors.splice(index, 1);
      renderMotors();
      markDirty("motors");
    }));
    body.append(row);
  });
  $("#motor-count").textContent = `${motors.length} 个电机`;
}

function modulePatch(module) {
  if (module === "servos" || module === "motors") {
    return { [MODULE_ROOTS[module]]: deepClone(state.config[MODULE_ROOTS[module]] || []) };
  }

  if (module === "usb_devices") {
    return { usb_devices: deepClone(state.config.usb_devices || {}) };
  }

  if (module === "asr") {
    const container = moduleContainer(module);
    const mode = $("#asr-mode").value;
    const selector = mode === "cloud" ? $("#asr-provider") : $("#asr-engine");
    const selected = selector.value;
    const patch = {
      asr: mode === "cloud"
        ? { mode, provider: selected, [selected]: {} }
        : { mode, engine: selected, [selected]: {} },
    };
    const panelSelector = mode === "cloud"
      ? `[data-asr-provider-panel="${selected}"]`
      : `[data-local-asr-engine-panel="${selected}"]`;
    $$(`${panelSelector} [data-path]`, container).forEach((input) => {
      if (input.dataset.secret === "true" && !input.value) return;
      if (input.dataset.optional === "true" && !input.value.trim()) return;
      setPath(patch, input.dataset.path, inputValue(input));
    });
    return patch;
  }

  const container = moduleContainer(module);
  const patch = {};
  $$('[data-path]', container).forEach((input) => {
    if (input.dataset.secret === "true" && !input.value) return;
    setPath(patch, input.dataset.path, inputValue(input));
  });
  return patch;
}

function refreshModuleFromSnapshot(module, payload) {
  const root = MODULE_ROOTS[module];
  setPath(state.config, root, deepClone(getPath(payload.config, root)));
  state.secretFields = payload.secret_fields || {};
  if (module === "asr") ensureAsrConfigs();
  if (module === "vad") ensureVadConfig();
  if (module === "llm") ensureLlmConfig();
  if (module === "hardware") ensureHardwareConfig();
  populateFields(moduleContainer(module));
  if (module === "asr") {
    updateAsrModePanels(state.config.asr.mode);
    updateAsrProviderPanels(state.config.asr.provider);
    updateLocalAsrEnginePanels(state.config.asr.engine);
  }
  if (module === "vad") updateVadProviderPanels(state.config.vad.provider);
  if (module === "hardware") updateHardwareBackendPanels(state.config.hardware.backend);
  if (module === "servos") renderServos();
  if (module === "motors") renderMotors();
  if (module === "usb_devices") renderUsbSelectors();
  $("#modified-at").textContent = `更新于 ${new Date(payload.modified_at).toLocaleString()}`;
}

async function loadConfig() {
  if (state.dirtyModules.size && !window.confirm("放弃尚未保存的模块修改并重新读取配置吗？")) return;
  state.loading = true;
  $("#reload-button").disabled = true;
  try {
    const payload = await api("/api/config");
    if (!payload.config || typeof payload.config !== "object" || Array.isArray(payload.config)) {
      throw new Error("配置服务没有返回有效的 config.yaml 内容");
    }
    state.config = payload.config;
    state.secretFields = payload.secret_fields || {};
    ensureRemoteControlConfig();
    ensureHardwareConfig();
    ensureVadConfig();
    ensureLlmConfig();
    ensureUsbDeviceConfig();
    ensureAsrConfigs();
    populateFields();
    updateAsrModePanels(state.config.asr.mode);
    updateAsrProviderPanels(state.config.asr.provider);
    updateLocalAsrEnginePanels(state.config.asr.engine);
    updateVadProviderPanels(state.config.vad.provider);
    updateHardwareBackendPanels(state.config.hardware.backend);
    renderServos();
    renderMotors();
    $("#config-path").textContent = payload.config_path;
    $("#config-path").title = payload.config_path;
    $("#modified-at").textContent = `更新于 ${new Date(payload.modified_at).toLocaleString()}`;
    setConfigControlsEnabled(true);
    setConnection(true, "配置已读取");
    clearAllDirty();
    clearModuleFeedbacks();
    $("#error-box").hidden = true;
    showToast("已读取 config.yaml 当前配置");
    loadUsbDevices({ quiet: true });
  } catch (error) {
    clearConfigurationView();
    setConfigControlsEnabled(false);
    if (error.status === 401) {
      error.details = ["请输入页面右上角的访问令牌，然后按 Enter 或点击“重新读取全部”。"];
      setConnection(false, "配置未读取：访问令牌无效或缺失");
      $("#access-token").focus();
    } else {
      setConnection(false, "配置读取失败");
    }
    showErrors(error);
    showToast(error.status === 401 ? "需要正确的访问令牌" : error.message, "error");
  } finally {
    state.loading = false;
    $("#reload-button").disabled = false;
  }
}

async function saveModule(module) {
  if (!MODULE_ROOTS[module]) return;
  const button = $(`[data-save-module="${module}"]`);
  if (!state.config) {
    const error = new Error("配置尚未读取，不能保存。请先输入正确的访问令牌并重新读取配置。");
    setModuleFeedback(module, "未保存：配置尚未读取", "error");
    showErrors(error);
    showToast(error.message, "error");
    return;
  }
  button.disabled = true;
  button.textContent = "保存中…";
  setModuleFeedback(module, "正在写入 config.yaml", "saving");
  $("#error-box").hidden = true;
  try {
    if (module === "asr") {
      const mode = $("#asr-mode").value;
      const selected = mode === "cloud" ? $("#asr-provider").value : $("#asr-engine").value;
      const panelSelector = mode === "cloud"
        ? `[data-asr-provider-panel="${selected}"]`
        : `[data-local-asr-engine-panel="${selected}"]`;
      const missingInput = $$(`${panelSelector} [required]`).find((input) => !input.value.trim());
      if (missingInput) {
        throw new Error(`请先填写${missingInput.closest("label")?.querySelector("span")?.textContent || "必填配置"}`);
      }
      if (mode === "cloud") {
        const secretInput = $(`[data-path="asr.${selected}.api_key"]`);
        const hasSavedSecret = Boolean(state.secretFields[`asr.${selected}.api_key`]);
        if (!secretInput.value && !hasSavedSecret) {
          throw new Error(`请先填写${$("#asr-provider").selectedOptions[0].textContent}的 API Key`);
        }
      }
    }
    const payload = await api("/api/config", {
      method: "POST",
      body: JSON.stringify({ patch: modulePatch(module) }),
    });
    refreshModuleFromSnapshot(module, payload);
    clearDirty(module);
    setConnection(true, "配置服务在线");
    setModuleFeedback(module, `已保存 ${new Date().toLocaleTimeString()}`, "success");
    showToast(payload.message || `${MODULE_LABELS[module]}模块已保存，重启主脑后生效`);
  } catch (error) {
    setModuleFeedback(module, "保存失败", "error");
    showErrors(error);
    showToast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "保存本模块";
  }
}

function bindEvents() {
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => {
    $$(".tab").forEach((item) => item.classList.toggle("active", item === tab));
    $$(".panel").forEach((panel) => panel.classList.toggle("active", panel.dataset.panel === tab.dataset.tab));
  }));
  $$('[data-path]').forEach((input) => input.addEventListener(input.type === "checkbox" ? "change" : "input", () => {
    markDirty(input.closest("[data-module]")?.dataset.module);
  }));
  $("#asr-mode").addEventListener("change", (event) => {
    if (state.config) state.config.asr.mode = event.target.value;
    updateAsrModePanels(event.target.value);
    markDirty("asr");
  });
  $("#asr-provider").addEventListener("change", (event) => {
    if (state.config) state.config.asr.provider = event.target.value;
    updateAsrProviderPanels(event.target.value);
    markDirty("asr");
  });
  $("#asr-engine").addEventListener("change", (event) => {
    if (state.config) state.config.asr.engine = event.target.value;
    updateLocalAsrEnginePanels(event.target.value);
    markDirty("asr");
  });
  $("#vad-provider").addEventListener("change", (event) => {
    if (state.config) state.config.vad.provider = event.target.value;
    updateVadProviderPanels(event.target.value);
    markDirty("vad");
  });
  $("#hardware-backend").addEventListener("change", (event) => {
    if (state.config) state.config.hardware.backend = event.target.value;
    updateHardwareBackendPanels(event.target.value);
    markDirty("hardware");
  });
  $$('[data-usb-role]').forEach((select) => {
    select.addEventListener("focus", () => loadUsbDevices({ quiet: true }));
    select.addEventListener("change", () => {
      const role = select.dataset.usbRole;
      if (select.value) state.config.usb_devices[role] = JSON.parse(select.value);
      else delete state.config.usb_devices[role];
      renderUsbSelectors();
      markDirty("usb_devices");
    });
  });
  $("#refresh-usb-devices").addEventListener("click", () => loadUsbDevices());
  $("#camera-preview-start").addEventListener("click", startCameraPreview);
  $("#camera-preview-stop").addEventListener("click", () => stopCameraPreview());
  $("#camera-preview-reconnect").addEventListener("click", reconnectCameraPreview);
  $("#change-token-button").addEventListener("click", changeAccessToken);
  $$('[data-save-module]').forEach((button) => button.addEventListener("click", () => saveModule(button.dataset.saveModule)));
  $("#reload-button").addEventListener("click", loadConfig);
  $("#access-token").addEventListener("input", () => {
    sessionStorage.setItem("waliConfigToken", getToken());
  });
  $("#access-token").addEventListener("change", () => {
    loadConfig();
  });
  $("#access-token").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      loadConfig();
    }
  });
  $("#add-servo").addEventListener("click", () => {
    const servos = state.config.servos || (state.config.servos = []);
    const ids = servos.map((item) => Number(item.id)).filter(Number.isFinite);
    servos.push({ id: ids.length ? Math.max(...ids) + 1 : 0, name: "new_servo", limit_1: 2000, limit_2: 6000, init: 4000 });
    renderServos();
    markDirty("servos");
  });
  $("#add-motor").addEventListener("click", () => {
    const motors = state.config.motors || (state.config.motors = []);
    const ids = motors.map((item) => Number(item.id)).filter(Number.isFinite);
    motors.push({ id: ids.length ? Math.max(...ids) + 1 : 0, name: "new_motor", max_speed: 100, neutral_speed: 0, invert_direction: false });
    renderMotors();
    markDirty("motors");
  });
  $("#close-error").addEventListener("click", () => { $("#error-box").hidden = true; });
  window.addEventListener("beforeunload", (event) => {
    if (!state.dirtyModules.size) return;
    event.preventDefault();
    event.returnValue = "";
  });
  window.addEventListener("pagehide", stopCameraPreviewOnPageExit);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && state.cameraPreview.active) stopCameraPreview({ quiet: true });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  $("#access-token").value = sessionStorage.getItem("waliConfigToken") || DEFAULT_ACCESS_TOKEN;
  prepareModuleFeedbacks();
  setConfigControlsEnabled(false);
  bindEvents();
  loadConfig();
});
