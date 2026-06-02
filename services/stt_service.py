# services/stt_service.py
import queue
import re
import sys
import threading
import time

import sherpa_onnx
import sounddevice as sd


class STTService:
    """Speech-to-text service with VAD endpoint detection."""
    SINGLE_ASCII_RE = re.compile(r"[A-Za-z]")
    MIXED_CJK_NOISE_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]{1,2}(?=[\u4e00-\u9fff])|(?<=[\u4e00-\u9fff])[A-Za-z]{1,2}(?![A-Za-z])")
    SINGLE_TOKEN_RE = re.compile(r"[A-Za-z]{1,2}")

    def __init__(self, model_dir=r"F:\well-e-bot\sherpa-onnx", on_sentence_received=None):
        self.model_dir = model_dir
        self.on_sentence_received = on_sentence_received
        self.is_running = False
        self.is_paused = False
        self.audio_stream = None
        self.audio_queue = queue.Queue(maxsize=80)
        self._stream_lock = threading.Lock()
        self._voice_state_lock = threading.Lock()
        self._listen_thread = None
        # Use audio energy as the main guard against short noise bursts that
        # otherwise get decoded into stray letters such as "A" or "Y".
        self.voice_rms_threshold = 0.015
        self.min_voiced_duration_sec = 0.35
        self.min_single_ascii_duration_sec = 0.55
        self._voiced_duration_sec = 0.0

        print("[STT] Loading local speech recognition engine...")
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{self.model_dir}/tokens.txt",
            encoder=f"{self.model_dir}/encoder-epoch-99-avg-1.int8.onnx",
            decoder=f"{self.model_dir}/decoder-epoch-99-avg-1.onnx",
            joiner=f"{self.model_dir}/joiner-epoch-99-avg-1.onnx",
            # sherpa-onnx requires modified_beam_search when hotwords are enabled.
            decoding_method="modified_beam_search",
            # Turn partial streaming text into final sentences after trailing silence.
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=20.0,
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            hotwords_file=f"{self.model_dir}/hotwords.txt",
            hotwords_score=2.5,
        )
        self.stream = self.recognizer.create_stream()

    def start(self):
        self.is_running = True

        self._listen_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._listen_thread.start()

        self.audio_stream = sd.InputStream(
            channels=1,
            dtype="float32",
            samplerate=16000,
            callback=self._audio_callback,
        )
        self.audio_stream.start()
        print("[STT] Speech listener started.")
        return True

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio callback only queues samples; sherpa stream is owned by decode thread."""
        if not self.is_running or self.is_paused:
            return

        samples = indata.copy().reshape(-1)
        if len(samples):
            rms = float((samples * samples).mean() ** 0.5)
            if rms >= self.voice_rms_threshold:
                self._mark_voiced(len(samples) / 16000.0)

        try:
            self.audio_queue.put_nowait(samples)
        except queue.Full:
            # Keep latency bounded by dropping the oldest audio chunk.
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_queue.put_nowait(samples)
            except queue.Full:
                pass

    def _drain_audio_queue(self):
        chunks = []
        while True:
            try:
                chunks.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break
        return chunks

    def _clear_audio_queue(self):
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def _mark_voiced(self, seconds):
        with self._voice_state_lock:
            self._voiced_duration_sec += seconds

    def _take_voiced_duration(self):
        with self._voice_state_lock:
            voiced_duration = self._voiced_duration_sec
            self._voiced_duration_sec = 0.0
        return voiced_duration

    def _clean_sentence(self, sentence):
        sentence = (sentence or "").strip()
        if not sentence:
            return ""

        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in sentence)
        if not has_cjk:
            return re.sub(r"\s+", " ", sentence).strip()

        sentence = self.MIXED_CJK_NOISE_RE.sub(" ", sentence)
        tokens = [
            token for token in sentence.split()
            if not self.SINGLE_TOKEN_RE.fullmatch(token)
        ]
        sentence = "".join(tokens) if tokens else sentence
        sentence = re.sub(r"\s+", " ", sentence).strip()
        return sentence

    def _should_publish_sentence(self, sentence, voiced_duration_sec):
        if not sentence:
            return False

        if voiced_duration_sec < self.min_voiced_duration_sec:
            return False

        if self.SINGLE_ASCII_RE.fullmatch(sentence):
            return voiced_duration_sec >= self.min_single_ascii_duration_sec

        return True

    def _process_loop(self):
        """Decode thread: the only thread that touches recognizer stream/decode/reset."""
        last_text = ""
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue

            sentence = None
            chunks = self._drain_audio_queue()

            with self._stream_lock:
                for samples in chunks:
                    self.stream.accept_waveform(16000, samples)

                while self.recognizer.is_ready(self.stream):
                    self.recognizer.decode_stream(self.stream)

                current_text = self.recognizer.get_result(self.stream)

                if current_text != last_text and current_text:
                    sys.stdout.write(f"\r[STT listening]: {current_text}")
                    sys.stdout.flush()
                    last_text = current_text

                if self.recognizer.is_endpoint(self.stream):
                    voiced_duration_sec = self._take_voiced_duration()
                    cleaned_sentence = self._clean_sentence(current_text)
                    if self._should_publish_sentence(cleaned_sentence, voiced_duration_sec):
                        sentence = cleaned_sentence
                    elif cleaned_sentence:
                        print(
                            f"\n[STT] Dropped noisy/short utterance: "
                            f"{cleaned_sentence!r} ({voiced_duration_sec:.2f}s voiced)"
                        )

                    self.recognizer.reset(self.stream)
                    last_text = ""

            if sentence:
                print("")
                if self.on_sentence_received:
                    try:
                        self.on_sentence_received(sentence)
                    except Exception as e:
                        print(f"[STT] Sentence callback failed: {e}")

            time.sleep(0.01)

    def pause(self):
        """Pause listening while the robot speaks."""
        self.is_paused = True
        self._clear_audio_queue()
        self._take_voiced_duration()

    def resume(self):
        """Resume listening with clean buffers."""
        self._clear_audio_queue()
        with self._stream_lock:
            self.recognizer.reset(self.stream)
        self._take_voiced_duration()
        self.is_paused = False

    def stop(self):
        self.is_running = False
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=1.0)
        print("[STT] Speech listener stopped.")
