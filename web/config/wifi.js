"use strict";
const element = (id) => document.getElementById(id);
const labels = {idle: "等待配网", starting: "正在准备", scanning: "正在搜索 Wi-Fi", setup: "等待选择 Wi-Fi", connecting: "正在连接 Wi-Fi", connected: "Wi-Fi 已连接", syncing: "正在同步 ESP32", ready: "配置成功", partial: "需要重试", failed: "操作失败"};
const busyStates = new Set(["starting", "scanning", "connecting", "connected", "syncing"]);
let pending = false;
let polling = false;
let scanning = false;
let pendingAction = "";
element("token").value = sessionStorage.getItem("waliConfigToken") || "";

function setPending(value) {
  pending = value;
  ["connect", "retry", "setup", "rescan"].forEach((id) => { element(id).disabled = value; });
  if (!value) pendingAction = "";
}
async function request(path, payload) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15000);
  const options = {credentials: "same-origin", signal: controller.signal, headers: {"X-Wali-Token": element("token").value.trim()}};
  if (payload !== undefined) {
    options.method = "POST";
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(payload);
  }
  try {
    const response = await fetch(`/api/host-network/${path}`, options);
    const result = await response.json();
    if (!response.ok || !result.ok) {
      const error = new Error(result.error || "操作失败");
      error.status = response.status;
      throw error;
    }
    return result;
  } finally { clearTimeout(timer); }
}
async function act(task) {
  try { await task(); }
  catch (error) {
    if (error.status === 401) {
      element("auth").hidden = false;
      element("token").focus();
      element("status").textContent = "配网会话未验证。请连接瓦力热点后重新打开此页，或输入设置页令牌再刷新。";
    } else {
      element("status").textContent = pending
        ? pendingAction === "rescan" ? "正在重新搜索附近 Wi-Fi，请稍后重新连接原来的瓦力热点，再刷新列表。" : "正在切换网络或同步设备。手机可能已断开热点，请查看胸前屏幕的新地址；无需重复提交。"
        : error.name === "AbortError" ? "读取超时，请稍后重新扫描。" : error.message;
    }
  }
}
async function refresh() {
  const status = await request("status");
  element("auth").hidden = status.hotspot_session === true;
  element("status").dataset.state = status.state;
  element("status").textContent = `${labels[status.state] || "正在读取"}：${status.detail || "请选择网络后输入密码。"}`;
  const address = element("new-address");
  address.hidden = !status.ip;
  if (status.ip) {
    address.href = `http://${status.ip}:${location.port || "80"}/wifi`;
    address.textContent = `配网地址：${address.href}`;
  }
  setPending(busyStates.has(status.state));
  return status;
}
async function refreshAndScan() {
  if (scanning) return;
  scanning = true;
  element("refresh").disabled = true;
  element("refresh").textContent = "正在读取和扫描…";
  try {
    const status = await refresh();
    if (busyStates.has(status.state)) return;
    const result = await request("scan");
    const selected = element("ssid").value;
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = result.networks.length ? "请选择附近的 Wi-Fi" : "列表为空，请点击下方重新搜索";
    element("scan-info").textContent = result.detail || (result.cached ? "显示最近搜索到的网络；未找到目标网络时可重新搜索。" : `找到 ${result.networks.length} 个 Wi-Fi`);
    const options = result.networks.map((network) => {
      const option = document.createElement("option");
      option.value = network.ssid;
      option.textContent = `${network.ssid} · ${network.signal}% · ${network.security || "开放网络"}`;
      return option;
    });
    element("ssid").replaceChildren(placeholder, ...options);
    if (result.networks.some((network) => network.ssid === selected)) element("ssid").value = selected;
  } finally {
    scanning = false;
    element("refresh").disabled = false;
    element("refresh").textContent = "刷新状态和 Wi-Fi 列表";
  }
}
async function submit(path, payload, message) {
  if (pending) return;
  setPending(true);
  pendingAction = path;
  try {
    await request(path, payload);
    element("status").textContent = message;
  } catch (error) { setPending(false); throw error; }
}
element("token").addEventListener("input", () => sessionStorage.setItem("waliConfigToken", element("token").value.trim()));
element("token").addEventListener("change", () => act(refreshAndScan));
element("refresh").addEventListener("click", () => act(refreshAndScan));
element("connect-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!element("ssid").value) { element("status").textContent = "请先从下拉框选择 Wi-Fi。"; return; }
  act(async () => {
    await submit("connect", {ssid: element("ssid").value, password: element("password").value}, "已提交，正在连接 Wi-Fi。手机会断开热点，请查看胸前屏幕的连接与同步状态。");
    element("password").value = "";
  });
});
element("setup").addEventListener("click", () => act(() => submit("setup", {}, "正在开启配网热点，请查看胸前屏幕。")));
element("rescan").addEventListener("click", () => act(() => submit("rescan", {}, "正在重新搜索 Wi-Fi，热点会短暂断开。请稍后重新连接原热点，再刷新列表。")));
element("retry").addEventListener("click", () => act(() => submit("retry-sync", {}, "正在重试 ESP32 同步，请稍候。")));
setInterval(async () => {
  if (!pending || polling) return;
  polling = true;
  const rescan = pendingAction === "rescan";
  try {
    await act(async () => {
      const status = await refresh();
      if (rescan && !busyStates.has(status.state)) await refreshAndScan();
    });
  } finally { polling = false; }
}, 4000);
act(refreshAndScan);
