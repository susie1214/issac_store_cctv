"""
사장님 얼굴 인식 모듈 — OpenCV DNN (dlib 불필요)
=================================================
등록된 얼굴(사장님/직원)은 도난 알람에서 제외.
OpenCV + 히스토그램 비교 방식으로 dlib 없이 동작.

사용법:
  python face_manager.py    ← 최초 1회 얼굴 등록
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

ENCODINGS_FILE = Path("models/face_encodings.pkl")
SIM_THRESHOLD  = 0.75   # 얼굴 유사도 임계값 (높을수록 엄격)

# OpenCV 내장 얼굴 감지 — 한글 경로 우회: C:/tmp 에 복사
import shutil
_CASCADE_SRC  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_CASCADE_PATH = "C:/tmp/haarcascade_frontalface_default.xml"
if not Path(_CASCADE_PATH).exists():
    Path("C:/tmp").mkdir(exist_ok=True)
    shutil.copy(_CASCADE_SRC, _CASCADE_PATH)


class FaceManager:
    def __init__(self):
        self.ready = True
        self._features: list[np.ndarray] = []
        self._names:    list[str]         = []

        self._detector = cv2.CascadeClassifier(_CASCADE_PATH)
        if self._detector.empty():
            print("[얼굴인식] Haar Cascade 로드 실패")
            self.ready = False
            return

        self._load()
        print(f"[얼굴인식] 준비 완료 | 등록 인원: {len(self._names)}")

    # ── 얼굴 특징 추출 (YCrCb 히스토그램) ────
    def _extract(self, face_img: np.ndarray) -> np.ndarray | None:
        if face_img.size == 0:
            return None
        resized = cv2.resize(face_img, (64, 64))
        ycrcb   = cv2.cvtColor(resized, cv2.COLOR_BGR2YCrCb)
        hist    = cv2.calcHist([ycrcb], [0, 1, 2], None,
                               [16, 16, 16], [0, 256] * 3)
        cv2.normalize(hist, hist)
        return hist.flatten()

    # ── 얼굴 감지 ─────────────────────────────
    def _detect_faces(self, frame: np.ndarray):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return faces  # [(x,y,w,h), ...]

    # ── 얼굴 등록 ─────────────────────────────
    def register(self, name: str, image_path: str) -> bool:
        img   = cv2.imread(image_path)
        if img is None:
            print(f"[얼굴인식] 이미지 로드 실패: {image_path}")
            return False
        faces = self._detect_faces(img)
        if len(faces) == 0:
            print(f"[얼굴인식] 얼굴 감지 실패: {image_path}")
            return False

        x, y, w, h = faces[0]
        feat = self._extract(img[y:y+h, x:x+w])
        if feat is None:
            return False

        self._features.append(feat)
        self._names.append(name)
        self._save()
        print(f"[얼굴인식] 등록 완료: {name}")
        return True

    # ── 직원 여부 판단 ────────────────────────
    def is_staff(self, frame: np.ndarray, bbox: tuple) -> bool:
        if not self.ready or not self._features:
            return False
        x1, y1, x2, y2 = bbox
        crop  = frame[y1:y2, x1:x2]
        faces = self._detect_faces(crop)
        if len(faces) == 0:
            return False

        fx, fy, fw, fh = faces[0]
        feat = self._extract(crop[fy:fy+fh, fx:fx+fw])
        if feat is None:
            return False

        for ref in self._features:
            score = float(np.dot(feat, ref) /
                          (np.linalg.norm(feat) * np.linalg.norm(ref) + 1e-8))
            if score >= SIM_THRESHOLD:
                return True
        return False

    # ── 저장/로드 ─────────────────────────────
    def _save(self):
        ENCODINGS_FILE.parent.mkdir(exist_ok=True)
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump({"names": self._names,
                         "features": self._features}, f)

    def _load(self):
        if not ENCODINGS_FILE.exists():
            return
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        self._names    = data.get("names", [])
        self._features = data.get("features", [])


# ── 최초 1회 얼굴 등록 ───────────────────────
if __name__ == "__main__":
    fm = FaceManager()
    if fm.ready:
        ok = fm.register("사장님", "9.jpg")
        print("등록 완료!" if ok else "등록 실패 — 사진에서 얼굴을 찾지 못했습니다.")
