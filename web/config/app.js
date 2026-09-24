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
    lastStatus: null,
    phaseKey: "",
    phaseStartedAt: 0,
    lastOperationalPhase: "launching",
  },
  choreography: {
    items: [],
    catalog: { channels: [], actions: [] },
    document: null,
    selected: null,
    pixelsPerSecond: 100,
    dirty: false,
    drag: null,
    loaded: false,
    audioObjectUrl: null,
    audioAssetId: null,
    audioPeaks: [],
    audioFrame: null,
    audioLoadToken: 0,
  },
};

const CAMERA_PREVIEW_POLL_MS = 180;
const CAMERA_PREVIEW_STATUS_POLL_MS = 400;

const CAMERA_PREVIEW_PHASES = Object.freeze({
  launching: { label: "启动 Web 采集", step: 0 },
  requesting_camera: { label: "请求 camera_capture_node", step: 1 },
  waiting_frame: { label: "启动摄像头并等待首帧", step: 2 },
  streaming: { label: "已收到 /camera_frame", step: 3 },
  stopping: { label: "正在释放摄像头", step: 0 },
  stopped: { label: "预览已停止", step: -1 },
});

const MODULE_ROOTS = Object.freeze({
  runtime: "launch",
  orchestration: "orchestration",
  mcp: "mcp",
  pipeline: "pipeline",
  dialog_motion: "dialog_motion",
  asr: "asr",
  wake_word: "wake_word",
  vad: "vad",
  audio_capture: "audio_capture",
  tts: "tts",
  llm: "llm",
  system_prompt: "system_prompt",
  hardware: "hardware",
  serial: "serial",
  i2c: "i2c",
  remote_control: "remote_control",
  vision: "vision",
  tft_preview: "tft_preview",
  servos: "servos",
  motors: "motors",
  usb_devices: "usb_devices",
});

const MODULE_LABELS = Object.freeze({
  runtime: "运行",
  mcp: "MCP 网关",
  pipeline: "对话链路",
  dialog_motion: "倾听表情",
  asr: "ASR",
  wake_word: "唤醒词",
  vad: "VAD",
  audio_capture: "WebRTC 人声增强",
  tts: "TTS",
  llm: "LLM",
  system_prompt: "系统提示词",
  hardware: "运动硬件后端",
  serial: "串口",
  i2c: "I²C",
  remote_control: "手柄遥控",
  vision: "视觉",
  tft_preview: "胸前屏幕预览",
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

const MCP_DEFAULTS = Object.freeze({
  enabled: false,
  host: "127.0.0.1",
  port: 5555,
  path: "/mcp",
  command_timeout_sec: 12,
});

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

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
  const reportedPhase = status.phase || (previewState === "running" ? "streaming" : previewState);
  if (CAMERA_PREVIEW_PHASES[reportedPhase] && !["stopping", "stopped"].includes(reportedPhase)) {
    state.cameraPreview.lastOperationalPhase = reportedPhase;
  }
  const operationalPhase = reportedPhase === "error"
    ? state.cameraPreview.lastOperationalPhase
    : reportedPhase;
  const phaseInfo = CAMERA_PREVIEW_PHASES[operationalPhase] || {
    label: reportedPhase || "未知阶段",
    step: -1,
  };
  const phaseKey = `${previewState}:${reportedPhase}`;
  if (reportedPhase !== "error" && phaseKey !== state.cameraPreview.phaseKey) {
    state.cameraPreview.phaseKey = phaseKey;
    state.cameraPreview.phaseStartedAt = Date.now();
  }
  state.cameraPreview.lastStatus = { ...status };
  const startingText = {
    launching: "正在启动采集进程",
    requesting_camera: "正在请求摄像头画面",
    waiting_frame: "等待 /camera_frame 首帧",
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
  $("#camera-preview-elapsed").textContent = state.cameraPreview.phaseStartedAt
    ? `${((Date.now() - state.cameraPreview.phaseStartedAt) / 1000).toFixed(1)}s`
    : "—";
  $("#camera-preview-phase").textContent = `${phaseInfo.label} (${reportedPhase || "unknown"})`;
  $("#camera-preview-source").textContent = status.source || "—";
  $("#camera-preview-device").textContent = status.device || "—";
  $("#camera-preview-resolution").textContent = status.width && status.height
    ? `${status.width} × ${status.height}`
    : "—";
  $("#camera-preview-fps").textContent = status.fps ? `${status.fps} FPS` : "—";
  $("#camera-preview-frame-age").textContent = Number.isFinite(status.frame_age_ms)
    ? `${status.frame_age_ms} ms`
    : "—";
  $("#camera-preview-diagnostic").textContent = status.diagnostic || status.error || "—";

  $$("#camera-preview-flow [data-camera-step]").forEach((step, index) => {
    step.classList.remove("complete", "active", "failed");
    if (phaseInfo.step < 0) return;
    if (index < phaseInfo.step || previewState === "running") step.classList.add("complete");
    else if (index === phaseInfo.step) {
      step.classList.add(previewState === "error" ? "failed" : "active");
    }
  });

  const placeholder = $("#camera-preview-placeholder");
  if (!placeholder.hidden && previewState !== "running") {
    placeholder.textContent = status.error || phaseInfo.label;
  }
}

function scheduleCameraPreviewPoll(delay = CAMERA_PREVIEW_POLL_MS) {
  clearTimeout(state.cameraPreview.timer);
  if (!state.cameraPreview.active) return;
  state.cameraPreview.timer = window.setTimeout(pollCameraPreviewFrame, delay);
}

async function pollCameraPreviewFrame() {
  if (!state.cameraPreview.active || state.cameraPreview.requesting) return;
  state.cameraPreview.requesting = true;
  let nextPollDelay = CAMERA_PREVIEW_POLL_MS;
  try {
    const shouldRefreshStatus = !state.cameraPreview.lastStatus
      || state.cameraPreview.lastStatus.state !== "running"
      || state.cameraPreview.frameCount % 8 === 0;
    const status = shouldRefreshStatus
      ? await api("/api/camera-preview/status")
      : state.cameraPreview.lastStatus;
    renderCameraPreviewStatus(status);
    if (["error", "stopped"].includes(status.state)) {
      state.cameraPreview.active = false;
      setCameraPreviewControls(false);
      clearCameraPreviewImage();
      $("#camera-preview-placeholder").hidden = false;
      if (status.state === "error") showToast(status.error || "摄像头预览失败", "error");
      return;
    }
    if (status.state !== "running") {
      nextPollDelay = CAMERA_PREVIEW_STATUS_POLL_MS;
      $("#camera-preview-placeholder").hidden = false;
      return;
    }

    const response = await fetch(`/api/camera-preview/frame?t=${Date.now()}`, {
      cache: "no-store",
      headers: apiHeaders(false),
    });
    if (!response.ok) {
      let payload = {};
      try { payload = await response.json(); } catch (_) { /* non-JSON error */ }
      if (response.status === 503 && ["starting", "running"].includes(payload.state)) {
        renderCameraPreviewStatus(payload);
        nextPollDelay = CAMERA_PREVIEW_STATUS_POLL_MS;
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
    scheduleCameraPreviewPoll(nextPollDelay);
  }
}

async function startCameraPreview() {
  if (state.cameraPreview.active) return;
  state.cameraPreview.active = true;
  state.cameraPreview.frameCount = 0;
  state.cameraPreview.lastStatus = null;
  state.cameraPreview.phaseKey = "";
  state.cameraPreview.phaseStartedAt = Date.now();
  state.cameraPreview.lastOperationalPhase = "launching";
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

function ensureAudioCaptureConfig() {
  const defaults = { webrtc_apm_enabled: true, webrtc_pre_gain_db: 6 };
  if (!state.config.audio_capture || typeof state.config.audio_capture !== "object") {
    state.config.audio_capture = {};
  }
  Object.entries(defaults).forEach(([key, value]) => {
    if (state.config.audio_capture[key] === undefined) state.config.audio_capture[key] = value;
  });
}

function ensureMcpConfig() {
  if (!state.config.mcp || typeof state.config.mcp !== "object" || Array.isArray(state.config.mcp)) {
    state.config.mcp = deepClone(MCP_DEFAULTS);
    return;
  }
  Object.entries(MCP_DEFAULTS).forEach(([key, value]) => {
    if (state.config.mcp[key] === undefined) state.config.mcp[key] = value;
  });
}

function updateWebRtcPreGainValue() {
  const slider = $("#webrtc-pre-gain");
  const output = $("#webrtc-pre-gain-value");
  if (slider && output) output.textContent = `${slider.value || 6} dB`;
}

function updateLlmAudioCapabilityNotice(mode) {
  const notice = $("#llm-audio-capability-notice");
  if (notice) notice.hidden = mode !== "multimodal";
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
  if ($("#mcp-token-output")) $("#mcp-token-output").value = "";
  if ($("#mcp-token-status")) $("#mcp-token-status").textContent = "配置尚未读取";
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
  if (module === "audio_capture") ensureAudioCaptureConfig();
  if (module === "llm") ensureLlmConfig();
  if (module === "hardware") ensureHardwareConfig();
  populateFields(moduleContainer(module));
  if (module === "asr") {
    updateAsrModePanels(state.config.asr.mode);
    updateAsrProviderPanels(state.config.asr.provider);
    updateLocalAsrEnginePanels(state.config.asr.engine);
  }
  if (module === "vad") updateVadProviderPanels(state.config.vad.provider);
  if (module === "audio_capture") updateWebRtcPreGainValue();
  if (module === "pipeline") updateLlmAudioCapabilityNotice(state.config.pipeline?.mode);
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
    ensureMcpConfig();
    ensureHardwareConfig();
    ensureVadConfig();
    ensureAudioCaptureConfig();
    ensureLlmConfig();
    ensureUsbDeviceConfig();
    ensureAsrConfigs();
    populateFields();
    updateAsrModePanels(state.config.asr.mode);
    updateAsrProviderPanels(state.config.asr.provider);
    updateLocalAsrEnginePanels(state.config.asr.engine);
    updateVadProviderPanels(state.config.vad.provider);
    updateWebRtcPreGainValue();
    updateLlmAudioCapabilityNotice(state.config.pipeline?.mode);
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
    loadMcpTokenStatus();
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

async function loadMcpTokenStatus() {
  const status = $("#mcp-token-status");
  if (!status) return;
  try {
    const payload = await api("/api/mcp-token/status");
    status.textContent = payload.configured ? "令牌已写入 config.yaml" : "尚未配置令牌";
  } catch (_) {
    status.textContent = "无法读取令牌状态";
  }
}

async function generateMcpToken() {
  const button = $("#generate-mcp-token-button");
  const output = $("#mcp-token-output");
  if (!button || !output) return;
  button.disabled = true;
  button.textContent = "生成中…";
  try {
    const payload = await api("/api/mcp-token/generate", { method: "POST", body: JSON.stringify({}) });
    output.value = payload.token;
    $("#mcp-token-status").textContent = "新令牌已写入 config.yaml；MCP 重启后使用新令牌";
    showToast(payload.message);
  } catch (error) {
    showErrors(error);
    showToast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "生成新令牌";
  }
}

function utf8ByteLength(value) {
  return new TextEncoder().encode(value).length;
}

function esp32NetworkPayload() {
  const wifi = [1, 2, 3].map((index) => ({
    ssid: $(`#esp32-wifi-ssid-${index}`).value,
    password: $(`#esp32-wifi-password-${index}`).value,
  }));
  wifi.forEach((entry, index) => {
    const number = index + 1;
    if (utf8ByteLength(entry.ssid) > 32) throw new Error(`Wi-Fi ${number} SSID 不能超过 32 个 UTF-8 字节`);
    if (utf8ByteLength(entry.password) > 64) throw new Error(`Wi-Fi ${number} 密码不能超过 64 个 UTF-8 字节`);
    if (!entry.ssid && entry.password) throw new Error(`未使用的 Wi-Fi ${number} 必须同时留空 SSID 和密码`);
  });
  if (!wifi.some((entry) => entry.ssid)) throw new Error("至少必须配置一个非空 SSID");
  const host = $("#esp32-network-host").value;
  if (!utf8ByteLength(host) || utf8ByteLength(host) > 64) throw new Error("图像服务器地址必须为 1–64 个 UTF-8 字节");
  const port = Number($("#esp32-network-port").value);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error("图像服务器端口必须是 1–65535 的整数");
  return { wifi, host, port };
}

function setEsp32NetworkStatus(text, kind = "") {
  const element = $("#esp32-network-status");
  element.textContent = text;
  element.dataset.state = kind;
}

function setEsp32NetworkBusy(busy, label = "") {
  const save = $("#esp32-network-save");
  const query = $("#esp32-network-query");
  save.disabled = busy;
  query.disabled = busy;
  save.textContent = busy ? label || "正在应用…" : "保存并应用";
}

async function saveEsp32Network() {
  let payload;
  try {
    payload = esp32NetworkPayload();
  } catch (error) {
    setEsp32NetworkStatus(error.message, "error");
    showToast(error.message, "error");
    return;
  }
  setEsp32NetworkBusy(true, "正在验证并应用…");
  setEsp32NetworkStatus("已发送 SET，正在等待 ESP32 验证 Wi-Fi、TCP 图像服务器和 HELLO（最多约 65 秒）。请勿关闭串口或刷新页面。", "saving");
  try {
    const result = await api("/api/esp32-network/save-and-apply", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    // Firmware never returns passwords; remove entered copies after success too.
    [1, 2, 3].forEach((index) => { $(`#esp32-wifi-password-${index}`).value = ""; });
    setEsp32NetworkStatus(`配置成功：ESP32 已建立本次 RAM 网络会话（SET #${result.set_seq}，APPLY #${result.apply_seq}）；上位机已保留私密配置，今后启动会重新下发。`, "success");
    showToast("ESP32 网络会话已成功建立");
  } catch (error) {
    setEsp32NetworkStatus(`配置失败：${error.message}`, "error");
    showToast(error.message, "error");
  } finally {
    setEsp32NetworkBusy(false);
  }
}

async function queryEsp32Network() {
  setEsp32NetworkBusy(true, "正在读取…");
  setEsp32NetworkStatus("正在通过 USB 串口读取 ESP32 当前网络状态…", "saving");
  try {
    const result = await api("/api/esp32-network/query", { method: "POST", body: JSON.stringify({}) });
    (result.wifi || []).forEach((entry, index) => {
      const input = $(`#esp32-wifi-ssid-${index + 1}`);
      if (input) input.value = entry.ssid || "";
    });
    $("#esp32-network-host").value = result.host || "";
    $("#esp32-network-port").value = result.port || "";
    // The firmware never returns passwords. Clear any old values so a newly
    // queried SSID can never accidentally be saved with a previous password.
    [1, 2, 3].forEach((index) => { $(`#esp32-wifi-password-${index}`).value = ""; });
    const selected = result.selected === 255 ? "未连接 Wi-Fi" : `Wi-Fi ${result.selected + 1}`;
    const flags = [result.active_from_nvs ? "旧版 NVS 配置" : "RAM 会话模式", result.candidate_present ? "有候选配置" : "无候选配置", result.apply_running ? "切换中" : "未切换"].join("；");
    setEsp32NetworkStatus(`已读取（${selected}，${flags}）。Wi-Fi 密码不会由设备返回，已清空，请在保存前重新输入。`, "success");
  } catch (error) {
    setEsp32NetworkStatus(`读取失败：${error.message}`, "error");
    showToast(error.message, "error");
  } finally {
    setEsp32NetworkBusy(false);
  }
}

function choreographyUid(prefix = "segment") {
  const random = globalThis.crypto?.randomUUID?.().replaceAll("-", "") || `${Date.now()}${Math.random()}`.replace(".", "");
  return `${prefix}-${random.slice(0, 10)}`;
}

function choreographySnap(value) {
  const step = Number($("#choreography-snap")?.value || 0.25);
  return Math.max(0, Math.round(Number(value) / step) * step);
}

function newChoreographyDocument() {
  return {
    schema_version: 1,
    id: "untitled_action",
    name: "未命名动作",
    start_pose: "neutral",
    timeline_seconds: 8,
    tracks: state.choreography.catalog.channels.map((channel) => ({ channel: channel.id, segments: [] })),
    actions: [],
    audio: null,
  };
}

function normalizeChoreographyDocument(document) {
  const copy = deepClone(document || newChoreographyDocument());
  copy.tracks = Array.isArray(copy.tracks) ? copy.tracks : [];
  copy.actions = Array.isArray(copy.actions) ? copy.actions : [];
  copy.audio = copy.audio && typeof copy.audio === "object" ? copy.audio : null;
  state.choreography.catalog.channels.forEach((channel) => {
    if (!copy.tracks.some((track) => track.channel === channel.id)) {
      copy.tracks.push({ channel: channel.id, segments: [] });
    }
  });
  return copy;
}

function choreographyTrack(channel) {
  return state.choreography.document?.tracks.find((track) => track.channel === channel);
}

function choreographyActionDefinition(name) {
  return state.choreography.catalog.actions.find((action) => action.id === name);
}

function choreographySegment(channel, id) {
  return choreographyTrack(channel)?.segments.find((segment) => segment.id === id);
}

function choreographySyncMetadata() {
  const document = state.choreography.document;
  if (!document) return;
  document.name = $("#choreography-name").value.trim();
  document.id = $("#choreography-id").value.trim();
  document.timeline_seconds = Number($("#choreography-duration").value);
}

function markChoreographyDirty() {
  state.choreography.dirty = true;
  const status = $("#choreography-status");
  if (status) status.textContent = "有未保存修改";
  const summary = $("#choreography-summary");
  if (summary) summary.textContent = "保存前会重新校验范围与重叠";
  $("#choreography-errors")?.setAttribute("hidden", "");
}

function renderChoreographySelect() {
  const select = $("#choreography-select");
  select.innerHTML = '<option value="">新动作</option>';
  state.choreography.items.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = `${item.name} · ${item.id}`;
    select.append(option);
  });
  const currentId = state.choreography.document?.id;
  if (state.choreography.items.some((item) => item.id === currentId)) select.value = currentId;
}

function renderChoreographyPalette() {
  const root = $("#choreography-action-palette");
  root.innerHTML = "";
  renderChoreographyAudioSelect();
  if (!state.choreography.catalog.actions.length) {
    root.innerHTML = '<p class="empty-copy">暂无 sequences.yaml 动作</p>';
    return;
  }
  state.choreography.catalog.actions.forEach((action) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "action-palette-item";
    button.draggable = true;
    button.dataset.sequenceName = action.id;
    button.innerHTML = `<strong>${escapeHtml(action.label)}</strong><small>${action.duration.toFixed(2)}s · ${escapeHtml(action.channels.join(" / ") || "事件")}</small>`;
    button.addEventListener("dragstart", (event) => {
      event.dataTransfer.effectAllowed = "copy";
      event.dataTransfer.setData("application/x-wali-sequence", action.id);
    });
    root.append(button);
  });
}

function renderChoreographyAudioSelect() {
  const select = $("#choreography-audio-select");
  select.innerHTML = '<option value="">不使用音乐</option>';
  (state.choreography.catalog.audio || []).forEach((asset) => {
    const option = document.createElement("option");
    option.value = asset.id;
    option.textContent = `${asset.name} · ${(Number(asset.size || 0) / 1024 / 1024).toFixed(1)}MB`;
    select.append(option);
  });
  select.value = state.choreography.document?.audio?.asset_id || "";
}

function formatChoreographyTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  return `${minutes}:${(safe % 60).toFixed(2).padStart(5, "0")}`;
}

function clearChoreographyAudioSource() {
  state.choreography.audioLoadToken += 1;
  const audio = $("#choreography-audio");
  audio.pause();
  audio.removeAttribute("src");
  audio.load();
  if (state.choreography.audioObjectUrl) URL.revokeObjectURL(state.choreography.audioObjectUrl);
  state.choreography.audioObjectUrl = null;
  state.choreography.audioAssetId = null;
  state.choreography.audioPeaks = [];
  cancelAnimationFrame(state.choreography.audioFrame);
  state.choreography.audioFrame = null;
  $("#choreography-audio-play").disabled = true;
  $("#choreography-audio-stop").disabled = true;
  $("#choreography-audio-play").textContent = "播放";
  $("#choreography-time-display").textContent = "0:00.00";
  $("#choreography-audio-hint").textContent = "添加音乐后，波形和播放头会成为整套动作的时间参照；不添加音乐也可以正常编排。";
}

async function decodeChoreographyWaveform(blob) {
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return [];
    const context = new AudioContextClass();
    const buffer = await context.decodeAudioData(await blob.arrayBuffer());
    const channels = Array.from(
      { length: buffer.numberOfChannels },
      (_, index) => buffer.getChannelData(index),
    );
    const bins = Math.min(12000, Math.max(600, Math.ceil(buffer.duration * 48)));
    const block = Math.max(1, Math.floor(buffer.length / bins));
    const peaks = [];
    for (let index = 0; index < bins; index += 1) {
      let minimum = 1;
      let maximum = -1;
      const start = index * block;
      const end = Math.min(buffer.length, start + block);
      const stride = Math.max(1, Math.floor((end - start) / 96));
      for (let sample = start; sample < end; sample += stride) {
        channels.forEach((channel) => {
          minimum = Math.min(minimum, channel[sample]);
          maximum = Math.max(maximum, channel[sample]);
        });
      }
      peaks.push([minimum, maximum]);
    }
    await context.close();
    return peaks;
  } catch (_) {
    return [];
  }
}

async function loadChoreographyAudio(assetId, { updateDocument = false } = {}) {
  if (!assetId) {
    clearChoreographyAudioSource();
    if (updateDocument && state.choreography.document) {
      state.choreography.document.audio = null;
      markChoreographyDirty();
      renderChoreographyTimeline();
    }
    return;
  }
  if (state.choreography.audioAssetId === assetId && state.choreography.audioObjectUrl) return;
  clearChoreographyAudioSource();
  const loadToken = state.choreography.audioLoadToken;
  const asset = (state.choreography.catalog.audio || []).find((item) => item.id === assetId);
  if (!asset) throw new Error("音乐资源不存在，请重新上传");
  const response = await fetch(`/api/choreography-audio/${encodeURIComponent(assetId)}`, {
    cache: "no-store",
    headers: apiHeaders(false),
  });
  if (!response.ok) throw new Error(`读取音乐失败 (${response.status})`);
  const blob = await response.blob();
  if (loadToken !== state.choreography.audioLoadToken) return;
  const objectUrl = URL.createObjectURL(blob);
  const audio = $("#choreography-audio");
  state.choreography.audioObjectUrl = objectUrl;
  state.choreography.audioAssetId = assetId;
  audio.src = objectUrl;
  const metadataReady = new Promise((resolve, reject) => {
    audio.addEventListener("loadedmetadata", resolve, { once: true });
    audio.addEventListener("error", () => reject(new Error("浏览器无法解码该音乐格式")), { once: true });
  });
  audio.load();
  try {
    await metadataReady;
  } catch (error) {
    if (loadToken !== state.choreography.audioLoadToken) return;
    throw error;
  }
  if (loadToken !== state.choreography.audioLoadToken) return;
  const peaks = await decodeChoreographyWaveform(blob);
  if (loadToken !== state.choreography.audioLoadToken) return;
  state.choreography.audioPeaks = peaks;
  if (updateDocument || !state.choreography.document.audio) {
    state.choreography.document.audio = {
      asset_id: assetId,
      name: asset.name,
      duration: Number(audio.duration.toFixed(3)),
    };
    const requiredTimeline = Math.ceil(audio.duration);
    if (requiredTimeline > Number(state.choreography.document.timeline_seconds)) {
      state.choreography.document.timeline_seconds = Math.min(600, requiredTimeline);
      $("#choreography-duration").value = state.choreography.document.timeline_seconds;
    }
    markChoreographyDirty();
  } else {
    state.choreography.document.audio.duration = Number(audio.duration.toFixed(3));
  }
  $("#choreography-audio-play").disabled = false;
  $("#choreography-audio-stop").disabled = false;
  $("#choreography-audio-hint").textContent = `${asset.name} · ${formatChoreographyTime(audio.duration)}`;
  renderChoreographyTimeline();
}

async function inspectChoreographyAudioFile(file) {
  const objectUrl = URL.createObjectURL(file);
  const audio = document.createElement("audio");
  audio.preload = "metadata";
  try {
    const duration = await new Promise((resolve, reject) => {
      audio.addEventListener("loadedmetadata", () => resolve(audio.duration), { once: true });
      audio.addEventListener("error", () => reject(new Error("浏览器无法解码该音乐格式")), { once: true });
      audio.src = objectUrl;
    });
    if (!Number.isFinite(duration) || duration < 0.1) throw new Error("无法读取音乐时长");
    if (duration > 600) throw new Error("音乐不能超过 10 分钟");
    return Number(duration.toFixed(3));
  } finally {
    audio.removeAttribute("src");
    audio.load();
    URL.revokeObjectURL(objectUrl);
  }
}

async function uploadChoreographyAudio(file) {
  if (!file) return;
  if (file.size > 64 * 1024 * 1024) {
    showToast("音乐文件不能超过 64MB", "error");
    return;
  }
  const targetDocument = state.choreography.document;
  const button = $("#choreography-audio-upload");
  button.disabled = true;
  button.textContent = "上传中…";
  try {
    const duration = await inspectChoreographyAudioFile(file);
    const payload = await api("/api/choreography-audio", {
      method: "POST",
      body: file,
      headers: {
        "Content-Type": file.type || "application/octet-stream",
        "X-Wali-Filename": encodeURIComponent(file.name),
        "X-Wali-Audio-Duration": String(duration),
      },
    });
    state.choreography.catalog.audio = [
      ...(state.choreography.catalog.audio || []).filter((item) => item.id !== payload.asset.id),
      payload.asset,
    ];
    renderChoreographyAudioSelect();
    if (state.choreography.document !== targetDocument) {
      showToast("音乐已上传，可在当前编排中选择");
      return;
    }
    $("#choreography-audio-select").value = payload.asset.id;
    await loadChoreographyAudio(payload.asset.id, { updateDocument: true });
    showToast(payload.message);
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "上传音乐";
    $("#choreography-audio-input").value = "";
  }
}

function createChoreographyWaveform(displayWidth, displayHeight = 108) {
  const peaks = state.choreography.audioPeaks;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("audio-waveform");
  svg.setAttribute("viewBox", `0 0 ${displayWidth} ${displayHeight}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("aria-hidden", "true");
  if (!peaks.length) return svg;

  const middle = displayHeight / 2;
  const amplitudes = peaks
    .map(([minimum, maximum]) => Math.max(Math.abs(minimum), Math.abs(maximum)))
    .sort((a, b) => a - b);
  const reference = amplitudes[Math.floor(amplitudes.length * 0.97)] || 1;
  const gain = middle * 0.82 / Math.max(0.04, reference);
  const envelope = [];
  peaks.forEach(([minimum, maximum]) => {
    envelope.push(Math.max(
      3,
      Math.min(middle * 0.9, Math.max(maximum, Math.abs(minimum)) * gain),
    ));
  });
  const smoothed = envelope.map((height, index) => (
    (envelope[index - 1] || height) * 0.2
    + height * 0.6
    + (envelope[index + 1] || height) * 0.2
  ));
  const xAt = (index) => displayWidth * index / Math.max(1, smoothed.length - 1);
  const top = smoothed.map((height, index) => `${xAt(index).toFixed(2)},${(middle - height).toFixed(2)}`);
  const bottom = smoothed
    .map((height, index) => `${xAt(index).toFixed(2)},${(middle + height).toFixed(2)}`)
    .reverse();
  const gradientId = `choreography-wave-${choreographyUid("gradient")}`;
  const defs = document.createElementNS(svg.namespaceURI, "defs");
  const gradient = document.createElementNS(svg.namespaceURI, "linearGradient");
  gradient.setAttribute("id", gradientId);
  gradient.setAttribute("x1", "0");
  gradient.setAttribute("x2", "1");
  gradient.setAttribute("y1", "0");
  gradient.setAttribute("y2", "0");
  [["0%", "#52e0ff"], ["48%", "#9b7cff"], ["100%", "#f67bd5"]].forEach(([offset, color]) => {
    const stop = document.createElementNS(svg.namespaceURI, "stop");
    stop.setAttribute("offset", offset);
    stop.setAttribute("stop-color", color);
    gradient.append(stop);
  });
  defs.append(gradient);
  svg.append(defs);

  const center = document.createElementNS(svg.namespaceURI, "line");
  center.setAttribute("x1", "0");
  center.setAttribute("x2", String(displayWidth));
  center.setAttribute("y1", String(middle));
  center.setAttribute("y2", String(middle));
  center.setAttribute("class", "audio-waveform-center");
  svg.append(center);

  const area = document.createElementNS(svg.namespaceURI, "path");
  area.setAttribute("d", `M ${top.join(" L ")} L ${bottom.join(" L ")} Z`);
  area.setAttribute("class", "audio-waveform-area");
  area.setAttribute("fill", `url(#${gradientId})`);
  svg.append(area);

  const contour = document.createElementNS(svg.namespaceURI, "path");
  contour.setAttribute("d", `M ${top.join(" L ")}`);
  contour.setAttribute("class", "audio-waveform-contour");
  svg.append(contour);
  return svg;
}

function updateChoreographyPlayhead() {
  const audio = $("#choreography-audio");
  const playhead = $("#choreography-playhead");
  if (playhead) playhead.style.left = `${audio.currentTime * state.choreography.pixelsPerSecond}px`;
  $("#choreography-time-display").textContent = formatChoreographyTime(audio.currentTime);
  if (!audio.paused && !audio.ended) {
    state.choreography.audioFrame = requestAnimationFrame(updateChoreographyPlayhead);
  } else {
    $("#choreography-audio-play").textContent = "播放";
    state.choreography.audioFrame = null;
  }
}

async function toggleChoreographyAudio() {
  const audio = $("#choreography-audio");
  if (!audio.src) return;
  if (audio.paused) {
    try {
      await audio.play();
      $("#choreography-audio-play").textContent = "暂停";
      cancelAnimationFrame(state.choreography.audioFrame);
      updateChoreographyPlayhead();
    } catch (error) {
      showToast(`音乐播放失败：${error.message}`, "error");
    }
  } else {
    audio.pause();
    $("#choreography-audio-play").textContent = "播放";
  }
}

function stopChoreographyAudio() {
  const audio = $("#choreography-audio");
  audio.pause();
  audio.currentTime = 0;
  updateChoreographyPlayhead();
}

function choreographyClientErrors() {
  const errors = [];
  const timeline = Number(state.choreography.document?.timeline_seconds || 0);
  state.choreography.document?.tracks.forEach((track) => {
    let previousEnd = 0;
    let position = 0;
    [...track.segments].sort((a, b) => a.start - b.start).forEach((segment) => {
      const end = Number(segment.start) + Number(segment.duration);
      if (Number(segment.start) < previousEnd - 0.0001) errors.push({ channel: track.channel, id: segment.id, message: "动作段重叠" });
      if (end > timeline + 0.0001) errors.push({ channel: track.channel, id: segment.id, message: "超出时间轴" });
      previousEnd = Math.max(previousEnd, end);
      position += Number(segment.delta);
      if (position < -100 || position > 100) errors.push({ channel: track.channel, id: segment.id, message: `累计位置 ${position}% 越界` });
    });
  });
  return errors;
}

function renderChoreographyInspector() {
  const selected = state.choreography.selected;
  const inspector = $("#choreography-inspector");
  if (!selected) {
    inspector.hidden = true;
    return;
  }
  const segment = choreographySegment(selected.channel, selected.id);
  if (!segment) {
    state.choreography.selected = null;
    inspector.hidden = true;
    return;
  }
  const channel = state.choreography.catalog.channels.find((item) => item.id === selected.channel);
  $("#segment-channel-label").textContent = channel?.label || selected.channel;
  $("#segment-start").value = segment.start;
  $("#segment-duration").value = segment.duration;
  $("#segment-delta").value = segment.delta;
  $("#segment-delta-output").textContent = `${segment.delta > 0 ? "+" : ""}${segment.delta}%`;
  inspector.hidden = false;
}

function addChoreographySegment(channel, start) {
  const track = choreographyTrack(channel);
  if (!track) return;
  const timeline = Number(state.choreography.document.timeline_seconds);
  const duration = Math.max(0.05, Math.min(1, timeline - start));
  if (duration <= 0.05 && start >= timeline) return;
  const segment = { id: choreographyUid(), start: choreographySnap(start), duration: choreographySnap(duration) || 0.25, delta: 20 };
  if (segment.start + segment.duration > timeline) segment.duration = Math.max(0.05, timeline - segment.start);
  track.segments.push(segment);
  state.choreography.selected = { channel, id: segment.id };
  markChoreographyDirty();
  renderChoreographyTimeline();
}

function renderChoreographyTimeline() {
  const root = $("#choreography-timeline");
  const documentModel = state.choreography.document;
  if (!root || !documentModel) return;
  const px = state.choreography.pixelsPerSecond;
  const timeline = Number(documentModel.timeline_seconds || 8);
  const contentWidth = Math.max(640, timeline * px);
  const errors = choreographyClientErrors();
  const invalid = new Set(errors.map((item) => `${item.channel}:${item.id}`));
  root.innerHTML = "";

  const ruler = document.createElement("div");
  ruler.className = "timeline-ruler";
  ruler.innerHTML = '<div class="timeline-corner">TRACK / TIME</div>';
  const scale = document.createElement("div");
  scale.className = "timeline-scale";
  scale.style.width = `${contentWidth}px`;
  scale.style.backgroundSize = `${px}px 100%`;
  scale.addEventListener("click", (event) => {
    const audio = $("#choreography-audio");
    if (!audio.src) return;
    audio.currentTime = Math.min(audio.duration || timeline, Math.max(0, (event.clientX - scale.getBoundingClientRect().left) / px));
    updateChoreographyPlayhead();
  });
  for (let second = 0; second <= timeline; second += 1) {
    const tick = document.createElement("span");
    tick.className = "timeline-tick";
    tick.style.left = `${second * px}px`;
    tick.textContent = `${second}s`;
    scale.append(tick);
  }
  ruler.append(scale);
  root.append(ruler);

  const audioRow = document.createElement("div");
  audioRow.className = "timeline-row audio-row";
  audioRow.innerHTML = '<div class="timeline-label">音乐<small>播放时钟</small></div>';
  const audioLane = document.createElement("div");
  audioLane.className = "timeline-lane";
  audioLane.style.width = `${contentWidth}px`;
  audioLane.style.backgroundSize = `${px}px 100%, 100% 22px`;
  audioLane.addEventListener("click", (event) => {
    const audio = $("#choreography-audio");
    if (!audio.src) return;
    audio.currentTime = Math.min(audio.duration || timeline, Math.max(0, (event.clientX - audioLane.getBoundingClientRect().left) / px));
    updateChoreographyPlayhead();
  });
  if (documentModel.audio) {
    const clip = document.createElement("div");
    clip.className = "audio-clip";
    const clipWidth = Math.max(24, Number(documentModel.audio.duration) * px);
    clip.style.width = `${clipWidth}px`;
    const waveform = createChoreographyWaveform(clipWidth);
    const label = document.createElement("span");
    label.className = "audio-clip-label";
    label.textContent = `${documentModel.audio.name} · ${formatChoreographyTime(documentModel.audio.duration)}`;
    clip.append(waveform, label);
    audioLane.append(clip);
  } else {
    const empty = document.createElement("div");
    empty.className = "audio-empty";
    empty.textContent = "未添加音乐 · 时间轴仍可独立使用";
    audioLane.append(empty);
  }
  audioRow.append(audioLane);
  root.append(audioRow);

  const actionRow = document.createElement("div");
  actionRow.className = "timeline-row action-row";
  actionRow.innerHTML = '<div class="timeline-label">已有动作<small>拖入</small></div>';
  const actionLane = document.createElement("div");
  actionLane.className = "timeline-lane";
  actionLane.style.width = `${contentWidth}px`;
  actionLane.style.backgroundSize = `${px}px 100%`;
  actionLane.addEventListener("dragover", (event) => { event.preventDefault(); event.dataTransfer.dropEffect = "copy"; });
  actionLane.addEventListener("drop", (event) => {
    event.preventDefault();
    const sequenceName = event.dataTransfer.getData("application/x-wali-sequence");
    const definition = choreographyActionDefinition(sequenceName);
    if (!definition) return;
    const start = choreographySnap((event.clientX - actionLane.getBoundingClientRect().left) / px);
    if (start + definition.duration > Number(documentModel.timeline_seconds)) {
      showToast("已有动作会超出时间轴，请延长时间轴或向前放置", "error");
      return;
    }
    documentModel.actions.push({ id: choreographyUid("action"), sequence_name: sequenceName, start });
    markChoreographyDirty();
    renderChoreographyTimeline();
  });
  documentModel.actions.forEach((action) => {
    const definition = choreographyActionDefinition(action.sequence_name);
    if (!definition) return;
    const clip = document.createElement("div");
    clip.className = "action-clip";
    clip.style.left = `${Number(action.start) * px}px`;
    clip.style.width = `${Math.max(28, definition.duration * px)}px`;
    clip.dataset.actionId = action.id;
    clip.innerHTML = `<span>${escapeHtml(definition.label)}</span><button type="button" title="删除">×</button>`;
    clip.querySelector("button").addEventListener("click", (event) => {
      event.stopPropagation();
      documentModel.actions = documentModel.actions.filter((item) => item.id !== action.id);
      markChoreographyDirty();
      renderChoreographyTimeline();
    });
    clip.addEventListener("pointerdown", (event) => startChoreographyDrag(event, { type: "action", id: action.id, originalStart: Number(action.start) }));
    actionLane.append(clip);
  });
  actionRow.append(actionLane);
  root.append(actionRow);

  state.choreography.catalog.channels.forEach((channel) => {
    const row = document.createElement("div");
    row.className = "timeline-row";
    row.innerHTML = `<div class="timeline-label">${escapeHtml(channel.label)}<small>${escapeHtml(channel.id)}</small></div>`;
    const lane = document.createElement("div");
    lane.className = "timeline-lane";
    lane.dataset.channel = channel.id;
    lane.style.width = `${contentWidth}px`;
    lane.style.backgroundSize = `${px}px 100%`;
    lane.addEventListener("click", (event) => {
      if (event.target !== lane) return;
      addChoreographySegment(channel.id, (event.clientX - lane.getBoundingClientRect().left) / px);
    });
    const track = choreographyTrack(channel.id);
    track?.segments.forEach((segment) => {
      const block = document.createElement("div");
      const isSelected = state.choreography.selected?.channel === channel.id && state.choreography.selected?.id === segment.id;
      block.className = `motion-segment ${segment.delta < 0 ? "negative" : "positive"}${invalid.has(`${channel.id}:${segment.id}`) ? " invalid" : ""}${isSelected ? " selected" : ""}`;
      block.style.left = `${Number(segment.start) * px}px`;
      block.style.width = `${Math.max(18, Number(segment.duration) * px)}px`;
      block.dataset.segmentId = segment.id;
      block.innerHTML = `<span>${segment.delta > 0 ? "+" : ""}${segment.delta}% · ${Number(segment.duration).toFixed(2)}s</span><i class="segment-resize"></i>`;
      block.addEventListener("click", (event) => {
        event.stopPropagation();
        state.choreography.selected = { channel: channel.id, id: segment.id };
        renderChoreographyTimeline();
      });
      block.addEventListener("pointerdown", (event) => startChoreographyDrag(event, {
        type: event.target.classList.contains("segment-resize") ? "resize" : "move",
        channel: channel.id,
        id: segment.id,
        originalStart: Number(segment.start),
        originalDuration: Number(segment.duration),
      }));
      lane.append(block);
    });
    row.append(lane);
    root.append(row);
  });
  const playhead = document.createElement("div");
  playhead.id = "choreography-playhead";
  playhead.className = "timeline-playhead";
  playhead.style.left = `${$("#choreography-audio").currentTime * px}px`;
  root.append(playhead);
  renderChoreographyInspector();
}

function startChoreographyDrag(event, drag) {
  if (event.button !== 0) return;
  event.preventDefault();
  event.stopPropagation();
  state.choreography.drag = { ...drag, startX: event.clientX };
}

function moveChoreographyDrag(event) {
  const drag = state.choreography.drag;
  if (!drag) return;
  const deltaSeconds = (event.clientX - drag.startX) / state.choreography.pixelsPerSecond;
  const timeline = Number(state.choreography.document.timeline_seconds);
  if (drag.type === "action") {
    const action = state.choreography.document.actions.find((item) => item.id === drag.id);
    const definition = action && choreographyActionDefinition(action.sequence_name);
    if (!action || !definition) return;
    action.start = Math.min(Math.max(0, choreographySnap(drag.originalStart + deltaSeconds)), Math.max(0, timeline - definition.duration));
  } else {
    const segment = choreographySegment(drag.channel, drag.id);
    if (!segment) return;
    if (drag.type === "move") {
      segment.start = Math.min(Math.max(0, choreographySnap(drag.originalStart + deltaSeconds)), Math.max(0, timeline - segment.duration));
    } else {
      segment.duration = Math.max(0.05, Math.min(30, choreographySnap(drag.originalDuration + deltaSeconds) || 0.05, timeline - segment.start));
    }
    state.choreography.selected = { channel: drag.channel, id: drag.id };
  }
  markChoreographyDirty();
  renderChoreographyTimeline();
}

function endChoreographyDrag() {
  state.choreography.drag = null;
}

function showChoreographyErrors(error) {
  const box = $("#choreography-errors");
  const details = error?.details?.length ? error.details : [error?.message || "动作编排无效"];
  box.innerHTML = `<strong>需要修改</strong><ul>${details.map((detail) => `<li>${escapeHtml(detail)}</li>`).join("")}</ul>`;
  box.hidden = false;
  $("#choreography-status").textContent = "校验未通过";
  $("#choreography-summary").textContent = `${details.length} 个问题`;
}

async function validateChoreography({ quiet = false } = {}) {
  choreographySyncMetadata();
  try {
    const payload = await api("/api/choreographies/validate", {
      method: "POST",
      body: JSON.stringify({ document: state.choreography.document }),
    });
    state.choreography.document = normalizeChoreographyDocument(payload.document);
    $("#choreography-errors").hidden = true;
    $("#choreography-status").textContent = "校验通过";
    $("#choreography-summary").textContent = `${payload.compiled.transitions.length} 段运动 · ${payload.compiled.actions.length} 个已有动作${payload.compiled.audio ? " · 1 条音乐" : ""}`;
    renderChoreographyTimeline();
    if (!quiet) showToast("动作编排校验通过");
    return payload;
  } catch (error) {
    showChoreographyErrors(error);
    if (!quiet) showToast(error.message, "error");
    throw error;
  }
}

async function saveChoreography() {
  choreographySyncMetadata();
  const button = $("#choreography-save");
  button.disabled = true;
  button.textContent = "保存中…";
  try {
    const payload = await api("/api/choreographies", {
      method: "POST",
      body: JSON.stringify({ document: state.choreography.document }),
    });
    state.choreography.document = normalizeChoreographyDocument(payload.document);
    state.choreography.dirty = false;
    const listing = await api("/api/choreographies");
    state.choreography.items = listing.items || [];
    state.choreography.catalog = listing.catalog || state.choreography.catalog;
    renderChoreographySelect();
    renderChoreographyTimeline();
    $("#choreography-status").textContent = "已保存";
    $("#choreography-summary").textContent = `${payload.compiled.transitions.length} 段运动 · ${payload.compiled.actions.length} 个已有动作${payload.compiled.audio ? " · 1 条音乐" : ""}`;
    $("#choreography-errors").hidden = true;
    showToast(payload.message);
  } catch (error) {
    showChoreographyErrors(error);
    showToast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "保存";
  }
}

function confirmDiscardChoreography() {
  return !state.choreography.dirty || window.confirm("放弃尚未保存的动作编排修改吗？");
}

function resetChoreographyEditor() {
  if (!confirmDiscardChoreography()) return;
  state.choreography.document = newChoreographyDocument();
  state.choreography.selected = null;
  state.choreography.dirty = false;
  clearChoreographyAudioSource();
  $("#choreography-select").value = "";
  renderChoreographyDocument();
}

function renderChoreographyDocument() {
  const documentModel = state.choreography.document;
  if (!documentModel) return;
  $("#choreography-name").value = documentModel.name || "未命名动作";
  $("#choreography-id").value = documentModel.id || "untitled_action";
  $("#choreography-duration").value = documentModel.timeline_seconds || 8;
  $("#choreography-errors").hidden = true;
  $("#choreography-status").textContent = state.choreography.dirty ? "有未保存修改" : "尚未校验";
  $("#choreography-summary").textContent = "从中性姿势开始";
  renderChoreographySelect();
  renderChoreographyAudioSelect();
  renderChoreographyTimeline();
  if (documentModel.audio?.asset_id) {
    loadChoreographyAudio(documentModel.audio.asset_id).catch((error) => {
      showToast(error.message, "error");
    });
  } else {
    clearChoreographyAudioSource();
  }
}

async function loadChoreography(choreographyId) {
  if (!choreographyId) {
    resetChoreographyEditor();
    return;
  }
  if (!confirmDiscardChoreography()) {
    renderChoreographySelect();
    return;
  }
  try {
    const payload = await api(`/api/choreographies/${encodeURIComponent(choreographyId)}`);
    state.choreography.document = normalizeChoreographyDocument(payload.document);
    state.choreography.selected = null;
    state.choreography.dirty = false;
    renderChoreographyDocument();
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function deleteChoreography() {
  const id = state.choreography.document?.id;
  if (!state.choreography.items.some((item) => item.id === id)) {
    showToast("当前动作尚未保存", "error");
    return;
  }
  if (!window.confirm(`确定删除动作“${state.choreography.document.name}”吗？`)) return;
  try {
    const payload = await api(`/api/choreographies/${encodeURIComponent(id)}`, { method: "DELETE" });
    state.choreography.items = state.choreography.items.filter((item) => item.id !== id);
    state.choreography.dirty = false;
    state.choreography.document = newChoreographyDocument();
    renderChoreographyDocument();
    showToast(payload.message);
  } catch (error) {
    showToast(error.message, "error");
  }
}

async function loadChoreographyWorkspace() {
  try {
    const payload = await api("/api/choreographies");
    state.choreography.items = payload.items || [];
    state.choreography.catalog = payload.catalog || { channels: [], actions: [] };
    state.choreography.loaded = true;
    state.choreography.document = normalizeChoreographyDocument(state.choreography.document || newChoreographyDocument());
    renderChoreographyPalette();
    renderChoreographyDocument();
  } catch (error) {
    $("#choreography-status").textContent = "无法读取动作编排";
    $("#choreography-summary").textContent = error.message;
  }
}

function updateSelectedSegment(field, value) {
  const selected = state.choreography.selected;
  const segment = selected && choreographySegment(selected.channel, selected.id);
  if (!segment) return;
  segment[field] = Number(value);
  markChoreographyDirty();
  renderChoreographyTimeline();
}

function bindEvents() {
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => {
    $$(".tab").forEach((item) => item.classList.toggle("active", item === tab));
    $$(".panel").forEach((panel) => panel.classList.toggle("active", panel.dataset.panel === tab.dataset.tab));
    if (tab.dataset.tab === "choreography" && !state.choreography.loaded) loadChoreographyWorkspace();
  }));
  $$('[data-path]').forEach((input) => input.addEventListener(input.type === "checkbox" ? "change" : "input", () => {
    markDirty(input.closest("[data-module]")?.dataset.module);
  }));
  $("#pipeline-mode").addEventListener("change", (event) => {
    updateLlmAudioCapabilityNotice(event.target.value);
  });
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
  $("#webrtc-pre-gain").addEventListener("input", () => updateWebRtcPreGainValue());
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
  $("#esp32-network-save").addEventListener("click", saveEsp32Network);
  $("#esp32-network-query").addEventListener("click", queryEsp32Network);
  $("#change-token-button").addEventListener("click", changeAccessToken);
  $$('[data-save-module]').forEach((button) => button.addEventListener("click", () => saveModule(button.dataset.saveModule)));
  $("#reload-button").addEventListener("click", loadConfig);
  $("#generate-mcp-token-button")?.addEventListener("click", generateMcpToken);
  $("#choreography-select").addEventListener("change", (event) => loadChoreography(event.target.value));
  $("#choreography-new").addEventListener("click", resetChoreographyEditor);
  $("#choreography-validate").addEventListener("click", () => validateChoreography());
  $("#choreography-save").addEventListener("click", saveChoreography);
  $("#choreography-delete").addEventListener("click", deleteChoreography);
  $("#choreography-audio-upload").addEventListener("click", () => $("#choreography-audio-input").click());
  $("#choreography-audio-input").addEventListener("change", (event) => uploadChoreographyAudio(event.target.files?.[0]));
  $("#choreography-audio-select").addEventListener("change", (event) => {
    loadChoreographyAudio(event.target.value, { updateDocument: true }).catch((error) => {
      showToast(error.message, "error");
    });
  });
  $("#choreography-audio-play").addEventListener("click", toggleChoreographyAudio);
  $("#choreography-audio-stop").addEventListener("click", stopChoreographyAudio);
  $("#choreography-audio").addEventListener("ended", updateChoreographyPlayhead);
  ["#choreography-name", "#choreography-id"].forEach((selector) => {
    $(selector).addEventListener("input", () => {
      choreographySyncMetadata();
      markChoreographyDirty();
    });
  });
  $("#choreography-duration").addEventListener("change", () => {
    choreographySyncMetadata();
    markChoreographyDirty();
    renderChoreographyTimeline();
  });
  $("#choreography-zoom").addEventListener("input", (event) => {
    state.choreography.pixelsPerSecond = Number(event.target.value);
    renderChoreographyTimeline();
  });
  $("#segment-start").addEventListener("change", (event) => updateSelectedSegment("start", event.target.value));
  $("#segment-duration").addEventListener("change", (event) => updateSelectedSegment("duration", event.target.value));
  $("#segment-delta").addEventListener("input", (event) => updateSelectedSegment("delta", event.target.value));
  $("#segment-remove").addEventListener("click", () => {
    const selected = state.choreography.selected;
    const track = selected && choreographyTrack(selected.channel);
    if (!track) return;
    track.segments = track.segments.filter((segment) => segment.id !== selected.id);
    state.choreography.selected = null;
    markChoreographyDirty();
    renderChoreographyTimeline();
  });
  window.addEventListener("pointermove", moveChoreographyDrag);
  window.addEventListener("pointerup", endChoreographyDrag);
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
    if (!state.dirtyModules.size && !state.choreography.dirty) return;
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
  loadChoreographyWorkspace();
});
