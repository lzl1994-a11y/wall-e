"""Stream local music through FFmpeg and derive compact spectrum bands."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from services.audio_output import OUTPUT_SAMPLE_RATE


SUPPORTED_SUFFIXES = frozenset({".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg"})
DEFAULT_MUSIC_DIRECTORY = Path(os.environ.get("WALI_MUSIC_DIR", "/root/wall-e/music")).expanduser()


def resolve_track(directory: str | Path, query: str = "") -> Path:
    """Resolve a display name inside the music directory; paths are never accepted."""
    root = Path(directory).expanduser()
    tracks = sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: path.name.casefold(),
    ) if root.is_dir() else []
    if not tracks:
        raise FileNotFoundError(f"音乐目录中没有支持的音频文件: {root}")
    name = str(query or "").strip()
    if not name:
        return tracks[0]
    if Path(name).name != name or any(separator in name for separator in ("/", "\\")):
        raise ValueError("歌曲只能使用文件名或标题，不能包含路径")
    wanted = name.casefold()
    exact = [path for path in tracks if wanted in {path.name.casefold(), path.stem.casefold()}]
    partial = [path for path in tracks if wanted in path.stem.casefold()]
    if exact or partial:
        return (exact or partial)[0]
    raise FileNotFoundError(f"没有找到歌曲: {name}")


class SpectrumAnalyzer:
    """Convert PCM chunks into smoothed logarithmic frequency bands."""

    def __init__(self, sample_rate: int = OUTPUT_SAMPLE_RATE, bands: int = 20) -> None:
        self.sample_rate = int(sample_rate)
        self.edges = np.geomspace(60.0, min(16_000.0, self.sample_rate / 2), bands + 1)
        self._levels = np.zeros(bands, dtype=np.float32)
        self._plan_size = 0
        self._fft_size = 0
        self._window = np.empty(0, dtype=np.float32)
        self._band_slices: list[tuple[int, int]] = []

    def _prepare(self, sample_count: int) -> None:
        if sample_count == self._plan_size:
            return
        self._plan_size = sample_count
        self._fft_size = 1 << max(10, (sample_count - 1).bit_length())
        self._window = np.hanning(sample_count).astype(np.float32)
        frequencies = np.fft.rfftfreq(self._fft_size, 1.0 / self.sample_rate)
        self._band_slices = [
            (
                int(np.searchsorted(frequencies, low, side="left")),
                int(np.searchsorted(frequencies, high, side="left")),
            )
            for low, high in zip(self.edges[:-1], self.edges[1:])
        ]

    def analyze(self, samples: np.ndarray) -> list[float]:
        pcm = np.asarray(samples, dtype=np.float32).reshape(-1)
        if pcm.size < 2:
            return self._levels.tolist()
        self._prepare(pcm.size)
        windowed = (pcm / 32768.0) * self._window
        magnitudes = (
            np.abs(np.fft.rfft(windowed, n=self._fft_size))
            / max(1.0, pcm.size / 2)
        )
        levels = np.zeros_like(self._levels)
        for index, (start, end) in enumerate(self._band_slices):
            selected = magnitudes[start:end]
            if selected.size:
                db = 20.0 * np.log10(float(selected.max()) + 1e-7)
                levels[index] = np.clip((db + 60.0) / 60.0, 0.0, 1.0)
        self._levels = np.maximum(levels, self._levels * 0.72)
        return self._levels.tolist()


class MusicPlayer:
    """One replaceable worker; sound hardware and TFT transport stay external."""

    def __init__(
        self,
        *,
        directory: str | Path = DEFAULT_MUSIC_DIRECTORY,
        on_audio: Callable[[np.ndarray], None],
        on_audio_end: Callable[[], None],
        on_spectrum: Callable[[list[float]], None],
        on_state: Callable[[str, str, str], None],
        sample_rate: int = OUTPUT_SAMPLE_RATE,
        chunk_ms: int = 100,
        spectrum_hz: float = 10.0,
        popen_factory=subprocess.Popen,
    ) -> None:
        self.directory = Path(directory).expanduser()
        self.on_audio = on_audio
        self.on_audio_end = on_audio_end
        self.on_spectrum = on_spectrum
        self.on_state = on_state
        self.sample_rate = int(sample_rate)
        self.chunk_samples = self.sample_rate * max(20, int(chunk_ms)) // 1000
        self._spectrum_every_chunks = max(
            1,
            round(self.sample_rate / self.chunk_samples / max(1.0, float(spectrum_hz))),
        )
        self._popen = popen_factory
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._speech_busy = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = None

    def play(self, query: str = "") -> Path:
        track = resolve_track(self.directory, query)
        self.stop()
        with self._lock:
            self._stop = threading.Event()
            self._thread = threading.Thread(
                target=self._run, args=(track, self._stop), name="music-player", daemon=True
            )
            self._thread.start()
        return track

    def stop(self) -> bool:
        with self._lock:
            thread, process, stop = self._thread, self._process, self._stop
        active = thread is not None and thread.is_alive()
        stop.set()
        if process is not None and process.poll() is None:
            process.terminate()
        if active and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        return active

    def set_speech_busy(self, busy: bool) -> None:
        if busy:
            self._speech_busy.set()
        else:
            self._speech_busy.clear()

    def _run(self, track: Path, stop: threading.Event) -> None:
        title = track.stem
        process = None
        failed = False
        self.on_state("loading", title, "")
        try:
            process = self._popen(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(track),
                    "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
                    "-ar", str(self.sample_rate), "-ac", "1", "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            with self._lock:
                if stop is self._stop:
                    self._process = process
            if process.stdout is None or process.stderr is None:
                raise RuntimeError("ffmpeg 音频管道创建失败")
            while self._speech_busy.is_set() and not stop.wait(0.05):
                pass
            if stop.is_set():
                return
            self.on_state("playing", title, "")
            analyzer = SpectrumAnalyzer(self.sample_rate)
            chunk_bytes = self.chunk_samples * 2
            spectrum_chunk = 0
            deadline = time.monotonic()
            while not stop.is_set():
                while self._speech_busy.is_set() and not stop.wait(0.05):
                    deadline = time.monotonic()
                if stop.is_set():
                    break
                data = process.stdout.read(chunk_bytes)
                if not data:
                    break
                usable = len(data) - len(data) % 2
                if not usable:
                    continue
                samples = np.frombuffer(data[:usable], dtype=np.int16).copy()
                self.on_audio(samples)
                if spectrum_chunk % self._spectrum_every_chunks == 0:
                    self.on_spectrum(analyzer.analyze(samples))
                spectrum_chunk += 1
                deadline += len(samples) / self.sample_rate
                stop.wait(max(0.0, deadline - time.monotonic()))

            if not stop.is_set():
                code = process.wait(timeout=2.0)
                if code:
                    error = process.stderr.read().decode("utf-8", errors="replace").strip()
                    raise RuntimeError(error or f"ffmpeg exited with code {code}")
        except Exception as exc:
            if not stop.is_set():
                failed = True
                self.on_state("error", title, str(exc))
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
            self.on_audio_end()
            self.on_spectrum([0.0] * 20)
            if not failed:
                self.on_state("stopped", title, "")
            with self._lock:
                if stop is self._stop:
                    self._process = None
                    self._thread = None


__all__ = ["DEFAULT_MUSIC_DIRECTORY", "MusicPlayer", "SpectrumAnalyzer", "resolve_track"]
