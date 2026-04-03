"""
소리 감지 모듈 — YAMNet TFLite (Google AudioSet)
=================================================
비명·유리깨짐·충돌음 등을 마이크로 실시간 감지.

준비:
  python download_models.py       # YAMNet 모델 다운로드
  pip install sounddevice         # 마이크 입력
  pip install tflite-runtime      # Orange Pi (ARM)
  pip install tensorflow          # x86/Windows 대안

동작:
  0.975 초 분량의 오디오를 YAMNet 에 입력 → 521 클래스 스코어
  관심 클래스(비명/유리깨짐 등) 스코어 > 임계값 → callback 호출

Orange Pi 권장:
  tflite-runtime (tensorflow 풀버전보다 훨씬 가벼움)
"""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import numpy as np

# ── 파라미터 ──────────────────────────────────────
SAMPLE_RATE  = 16000              # YAMNet 요구 샘플레이트
CHUNK_SAMP   = 15600              # 0.975 초 × 16000 Hz
INFER_SEC    = 1.0                # 추론 주기 (초)
SCORE_TH     = 0.25              # 감지 임계값

# 감지 대상 YAMNet 클래스명 → 한글 표시명
ALERT_LABELS: dict[str, str] = {
    "Screaming":           "비명",
    "Shout, shout":        "고함",
    "Shout":               "고함",
    "Breaking":            "유리깨짐",
    "Glass":               "유리소리",
    "Crash":               "충돌음",
    "Bang":                "충격음",
    "Gunshot, gunfire":    "총성",
    "Explosion":           "폭발음",
    "Alarm":               "경보음",
    "Smoke detector":      "화재경보",
}


class SoundDetector:
    """
    사용법:
        def on_sound(label_kr, score):
            print(f"소리 감지: {label_kr} ({score:.2f})")

        sd = SoundDetector(Path("models/yamnet.tflite"), callback=on_sound)
        sd.start()
        ...
        sd.stop()
    """

    def __init__(self, model_path: Path, callback=None):
        """
        model_path : models/yamnet.tflite 경로
        callback   : fn(label_kr: str, score: float) — 감지 시 호출
        """
        self.ready    = False
        self.callback = callback
        self._running = False
        self._thread: threading.Thread | None = None
        self._classes: list[str] = []

        if not model_path.exists():
            print(f"[소리감지] 모델 없음: {model_path}")
            print("  → python download_models.py 실행 후 재시작")
            return

        # ── TFLite 인터프리터 로드 ──────────────────
        try:
            try:
                import tflite_runtime.interpreter as tflite   # Orange Pi 경량
            except ImportError:
                import tensorflow as tf                        # x86/Windows
                tflite = tf.lite
            self._interp = tflite.Interpreter(str(model_path))
            self._interp.allocate_tensors()
            self._inp  = self._interp.get_input_details()[0]
            self._outp = self._interp.get_output_details()
            print("[소리감지] YAMNet TFLite 로드 ✅")
        except Exception as e:
            print(f"[소리감지] 모델 로드 실패: {e}")
            return

        # ── 클래스 목록 로드 ────────────────────────
        csv_path = model_path.parent / "yamnet_class_map.csv"
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self._classes = [row["display_name"] for row in reader]
            print(f"[소리감지] 클래스 목록 로드: {len(self._classes)}종")
        else:
            print(f"[소리감지] 클래스 CSV 없음: {csv_path}  (python download_models.py 재실행)")
            return

        self.ready = True

    # ── 외부 인터페이스 ───────────────────────────
    def start(self):
        if not self.ready:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[소리감지] 마이크 모니터링 시작")

    def stop(self):
        self._running = False

    # ── 추론 ─────────────────────────────────────
    def _infer(self, audio: np.ndarray) -> list[tuple[str, str, float]]:
        """
        audio : float32, 16kHz, mono, -1~1 범위, 길이 CHUNK_SAMP
        반환  : [(yamnet_label, 한글명, score), ...]
        """
        wav = audio.astype(np.float32)
        # 길이 맞추기
        if len(wav) < CHUNK_SAMP:
            wav = np.pad(wav, (0, CHUNK_SAMP - len(wav)))
        else:
            wav = wav[:CHUNK_SAMP]

        self._interp.set_tensor(self._inp["index"], wav)
        self._interp.invoke()

        # output[0]: scores (frames, 521) / output[1]: embeddings / output[2]: log_mel
        scores = self._interp.get_tensor(self._outp[0]["index"])
        mean_scores = scores.mean(axis=0)   # 시간 프레임 평균

        hits = []
        for idx, score in enumerate(mean_scores):
            if score < SCORE_TH or idx >= len(self._classes):
                continue
            label = self._classes[idx]
            if label in ALERT_LABELS:
                hits.append((label, ALERT_LABELS[label], float(score)))

        return hits

    # ── 메인 루프 (별도 스레드) ──────────────────
    def _loop(self):
        try:
            import sounddevice as sd
        except ImportError:
            print("[소리감지] sounddevice 없음: pip install sounddevice")
            return

        # 롤링 버퍼 — 항상 최신 CHUNK_SAMP 개 샘플 유지
        buf = np.zeros(CHUNK_SAMP, dtype=np.float32)

        def _audio_cb(indata, frames, time_info, status):
            nonlocal buf
            chunk = indata[:, 0] if indata.ndim > 1 else indata.ravel()
            n = len(chunk)
            buf = np.roll(buf, -n)
            buf[-n:] = chunk

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=2048,
                callback=_audio_cb,
            ):
                while self._running:
                    time.sleep(INFER_SEC)
                    hits = self._infer(buf.copy())
                    for en_label, kr_label, score in hits:
                        if self.callback:
                            self.callback(kr_label, score)
        except Exception as e:
            print(f"[소리감지] 마이크 오류: {e}")
            print("  → 마이크 연결 확인 또는 sounddevice 재설치")
