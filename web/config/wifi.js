"use strict";
const element = (id) => document.getElementById(id);
let pending = false;
async function request(path, payload) {
  const options = {headers: {"X-Wali-Token": element("token").value}};
  if (payload !== undefined) {
    options.method = "POST";
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(payload);
  }
  const response = await fetch(`/api/host-network/${path}`, options);
  const result = await response.json();
  if (!response.ok || !result.ok) throw new Error(result.error || "操作失败");
  return result;
}
async function refresh() {
  const status = await request("status");
  element("status").textContent = `${status.state}：${status.detail || ""} ${status.ip ? `地址 http://${status.ip}:${location.port || "80"}/wifi` : ""}`;
  pending = ["starting", "connecting", "syncing"].includes(status.state);
}
async function act(task) {
  try { await task(); }
  catch (error) { element("status").textContent = error.message; }
}
element("refresh").addEventListener("click", () => act(async () => {
  await refresh();
  const result = await request("scan");
  element("networks").replaceChildren(...result.networks.map((network) => {
    const option = document.createElement("option");
    option.value = network.ssid;
    option.label = `${network.signal}% ${network.security}`;
    return option;
  }));
}));
element("connect-form").addEventListener("submit", (event) => {
  event.preventDefault();
  if (pending) return;
  pending = true;
  act(async () => {
    try {
      await request("connect", {ssid: element("ssid").value, password: element("password").value});
      element("password").value = "";
      element("status").textContent = "已提交，正在切换 Wi-Fi。手机可能断开，请查看瓦力胸前屏幕的新地址；失败时重新连接原热点。";
    } catch (error) { pending = false; throw error; }
  });
});
element("setup").addEventListener("click", () => act(async () => {
  await request("setup", {});
  pending = true;
  element("status").textContent = "正在开启热点，请查看瓦力胸前屏幕并连接显示的热点。";
}));
element("retry").addEventListener("click", () => act(async () => { await request("retry-sync", {}); pending = true; }));
setInterval(() => {
  if (pending) act(refresh);
}, 4000);
