#!/usr/bin/env python3
"""Generate three separately synthesized Pipa Xing segments as one WAV file."""

import argparse
from datetime import datetime
import threading
import time
import wave
from pathlib import Path

import numpy as np

from services.audio_output import OUTPUT_SAMPLE_RATE
from services.audio_silence import TurnAudioTrimmer
from services.tts_pipeline import OrderedTTSPipeline
from services.tts_service import TTSService


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "pipa_tts_3_segments.wav"
DEFAULT_TRIMMED_OUTPUT = ROOT / "pipa_tts_3_segments_trimmed.wav"
LINES = (
    "浔阳江头夜送客，枫叶荻花秋瑟瑟。",
    "主人下马客在船，举酒欲饮无管弦。",
    "醉不成欢惨将别，别时茫茫江浸月。",
)


def boundary_silence_ms(samples: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """Estimate leading/trailing near-silence while preserving quiet phonemes."""
    if samples.size == 0:
        return 0.0, 0.0
    magnitude = np.abs(samples.astype(np.int32))
    peak = int(magnitude.max())
    threshold = max(96, int(peak * 0.005))
    audible = np.flatnonzero(magnitude > threshold)
    if audible.size == 0:
        duration_ms = samples.size * 1000.0 / sample_rate
        return duration_ms, duration_ms
    leading_ms = int(audible[0]) * 1000.0 / sample_rate
    trailing_ms = (samples.size - int(audible[-1]) - 1) * 1000.0 / sample_rate
    return leading_ms, trailing_ms


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(target: Path) -> None:
        with target.open("wb") as output_file:
            with wave.open(output_file, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples.astype(np.int16, copy=False).tobytes())

    try:
        write(path)
        return path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        write(fallback)
        print(f"target_locked={path} fallback={fallback}")
        return fallback


def generate(args) -> Path:
    service = TTSService(
        voice=args.voice,
        rate=args.rate,
        pitch=args.pitch,
        sample_rate=args.sample_rate,
    )
    completed = threading.Event()
    segments = []
    errors = []

    pipeline = OrderedTTSPipeline(
        synthesize=service.synthesize,
        on_audio=lambda samples, text, elapsed: segments.append(
            (text, samples, elapsed)
        ),
        on_turn_end=lambda _turn_id: completed.set(),
        on_error=lambda text, error, elapsed: errors.append(
            (text, error, elapsed)
        ),
        workers=args.workers,
    )

    try:
        for line in LINES:
            pipeline.submit_speech(line)
        pipeline.submit_turn_end("pipa-probe")
        pipeline.shutdown()
    finally:
        # Let Edge-TTS close its HTTP transports before stopping the event loop.
        time.sleep(0.25)
        service.shutdown()

    if errors:
        details = "; ".join(
            f"{text}: {error} ({elapsed:.2f}s)"
            for text, error, elapsed in errors
        )
        raise RuntimeError(f"TTS synthesis failed: {details}")
    if not completed.is_set() or len(segments) != len(LINES):
        raise RuntimeError("TTS pipeline did not produce all ordered segments")

    combined = np.concatenate([samples for _, samples, _ in segments])
    output = write_wav(Path(args.output).resolve(), combined, args.sample_rate)
    trimmer = TurnAudioTrimmer(
        sample_rate=args.sample_rate,
        keep_silence_ms=args.keep_silence_ms,
        threshold_dbfs=args.threshold_dbfs,
    )
    trim_results = [trimmer.process(samples) for _, samples, _ in segments]
    trimmed = np.concatenate([result.samples for result in trim_results])
    trimmed_output = write_wav(
        Path(args.trimmed_output).resolve(),
        trimmed,
        args.sample_rate,
    )

    offset_ms = 0.0
    for index, ((text, samples, elapsed), trim_result) in enumerate(
        zip(segments, trim_results),
        1,
    ):
        duration_ms = samples.size * 1000.0 / args.sample_rate
        leading_ms, trailing_ms = boundary_silence_ms(samples, args.sample_rate)
        print(
            f"segment={index} start={offset_ms:.1f}ms duration={duration_ms:.1f}ms "
            f"leading_silence={leading_ms:.1f}ms trailing_silence={trailing_ms:.1f}ms "
            f"trimmed={trim_result.processed_ms:.1f}ms "
            f"cut_head={trim_result.leading_cut_ms:.1f}ms "
            f"cut_tail={trim_result.trailing_cut_ms:.1f}ms "
            f"synthesis={elapsed:.2f}s text={text}"
        )
        offset_ms += duration_ms

    print(
        f"output={output} sample_rate={args.sample_rate} channels=1 "
        f"sample_width=16bit duration={combined.size / args.sample_rate:.3f}s"
    )
    print(
        f"trimmed_output={trimmed_output} sample_rate={args.sample_rate} channels=1 "
        f"sample_width=16bit duration={trimmed.size / args.sample_rate:.3f}s"
    )
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--trimmed-output", default=str(DEFAULT_TRIMMED_OUTPUT))
    parser.add_argument("--voice", default="zh-CN-YunxiaNeural")
    parser.add_argument("--rate", default="+20%")
    parser.add_argument("--pitch", default="+5Hz")
    parser.add_argument("--sample-rate", type=int, default=OUTPUT_SAMPLE_RATE)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--keep-silence-ms", type=float, default=100.0)
    parser.add_argument("--threshold-dbfs", type=float, default=-45.0)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
