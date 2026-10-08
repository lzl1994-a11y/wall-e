"""Per-session timing counters; never retain a datagram or its contents."""
import time


class TransportDiagnostics:
    def __init__(self, transport):
        self.transport = transport
        self.reset()
        ice = transport.transport
        original_recv = ice._recv
        original_next = transport._recv_next
        original_send = transport._send_rtp
        self._received_at = None

        async def recv():
            data = await original_recv()
            self._received_at = time.monotonic()
            self._received_kind = self.kind(data)
            queue = getattr(ice._connection, "_queue", None)
            if queue is not None:
                self.queue_peak = max(self.queue_peak, queue.qsize())
            return data

        async def recv_next():
            self._received_at = None
            try:
                return await original_next()
            finally:
                if self._received_at is not None:
                    self.record("recv_" + self._received_kind, self._received_at)

        async def send(data):
            started = time.monotonic()
            try:
                return await original_send(data)
            finally:
                self.record("send_" + self.kind(data), started)

        ice._recv = recv
        transport._recv_next = recv_next
        transport._send_rtp = send

    @staticmethod
    def kind(data):
        if 19 < data[0] < 64:
            return "dtls"
        if 127 < data[0] < 192:
            return "rtcp" if 192 <= data[1] <= 208 else "rtp"
        return "other"

    def reset(self):
        self.counters = {}
        self.queue_peak = 0

    def record(self, kind, started):
        elapsed = (time.monotonic() - started) * 1000
        entry = self.counters.setdefault(kind, [0, 0.0, 0.0])
        entry[0] += 1
        entry[1] += elapsed
        entry[2] = max(entry[2], elapsed)

    def snapshot(self):
        return {"ice_queue_peak": self.queue_peak, "stages": {
            kind: {"count": entry[0], "total_ms": round(entry[1], 2),
                   "max_ms": round(entry[2], 2)}
            for kind, entry in self.counters.items()}}


def install_transport_diagnostics(peer):
    transports = {transceiver.receiver.transport for transceiver in peer.getTransceivers()}
    return [TransportDiagnostics(transport) for transport in transports]
