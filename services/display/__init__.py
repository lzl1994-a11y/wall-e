"""TFT and virtual-display protocols, sessions, rendering, and state.

English
-------
Display services define preview request/result messages, TCP framing, TFT text
rendering, virtual-display state, and bridge/client/server behavior.  Callers
submit semantic content or prepared frames; this package owns connection and
encoding details.  It does not choose dialog answers or game actions.  A new
screen backend should implement this boundary without changing its producers.

中文
----
显示层定义预览请求与结果消息、TCP 帧协议、TFT 文本渲染、虚拟显示状态以及桥接、客户端、
服务端行为。调用方只提交语义内容或已准备好的画面，连接和编码细节由本包负责；本层不决定
对话答案或游戏动作。未来更换屏幕后端时，应实现同一显示边界，而不是修改所有内容生产者。
"""
