"""
도난 의심 행동 감지 모듈 — Gemini VLM
========================================
행거/진열대 zone에 N초 이상 머문 사람을 Gemini로 분석.
"옷/물건을 몸에 숨기거나 가방에 넣는 행동"이면 알람.

흐름:
  YOLO bbox → zone 체류 10초 이상 → Gemini 판단 → 카카오 알림
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL      = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent?key=" + GEMINI_API_KEY
)

DWELL_TH_SEC    = 10    # zone 체류 임계값 (초)
COOLDOWN_SEC    = 60    # 동일 ID 재판단 억제 (초)
GEMINI_INTERVAL = 5     # Gemini 호출 간격 (초) — API 비용 절약

PROMPT = """이 CCTV 영상 캡처를 분석해주세요.

다음 행동이 보이면 'YES', 아니면 'NO'만 답하세요:
- 옷이나 물건을 몸속(옷 안쪽, 가방, 주머니)에 숨기는 행동
- 진열대에서 물건을 집어 빠르게 숨기는 행동
- 주변을 두리번거리며 물건을 감추는 행동

단순히 물건을 구경하거나 고르는 행동은 NO입니다.
반드시 YES 또는 NO 한 단어만 답하세요."""


class TheftDetector:
    """
    사용법:
        td = TheftDetector(zone_pts=[[x,y],...])
        result = td.update(tid, cx, cy, frame, is_staff)
        # result: None=정상, dict=도난의심({"tid":..,"frame":..})
    """

    def __init__(self, zone_pts: list = None):
        self.zone_pts  = zone_pts or []   # 행거/진열대 구역
        self._dwell:   dict[int, float] = {}   # tid → zone 진입 시각
        self._last_cd: dict[int, float] = {}   # tid → 마지막 알람 시각
        self._last_gemini = 0.0

        if not GEMINI_API_KEY:
            print("[도난감지] .env 에 GEMINI_API_KEY 없음")

    # ── zone 안에 있는지 판단 ─────────────────
    def _in_zone(self, cx: int, cy: int) -> bool:
        if len(self.zone_pts) < 3:
            return False
        pts = np.array(self.zone_pts, dtype=np.int32)
        return cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0

    # ── Gemini 판단 ───────────────────────────
    def _ask_gemini(self, frame: np.ndarray) -> bool:
        if not GEMINI_API_KEY:
            return False

        # API 호출 간격 제한
        now = time.time()
        if now - self._last_gemini < GEMINI_INTERVAL:
            return False
        self._last_gemini = now

        # 이미지 → base64
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buf.tobytes()).decode()

        payload = {
            "contents": [{
                "parts": [
                    {"text": PROMPT},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            }]
        }
        try:
            res = requests.post(GEMINI_URL, json=payload, timeout=10)
            if res.status_code == 200:
                answer = (res.json()
                          ["candidates"][0]["content"]["parts"][0]["text"]
                          .strip().upper())
                print(f"[도난감지] Gemini 판단: {answer}")
                return answer.startswith("YES")
        except Exception as e:
            print(f"[도난감지] Gemini 오류: {e}")
        return False

    # ── 매 프레임 호출 ────────────────────────
    def update(self, tid: int, cx: int, cy: int,
               frame: np.ndarray, is_staff: bool = False
               ) -> dict | None:
        """
        Returns:
          None   → 이상 없음
          dict   → 도난 의심 {"tid": int, "frame": np.ndarray}
        """
        if is_staff:
            self._dwell.pop(tid, None)
            return None

        now = time.time()

        if self._in_zone(cx, cy):
            # zone 진입 시각 기록
            if tid not in self._dwell:
                self._dwell[tid] = now

            dwell = now - self._dwell[tid]

            # N초 이상 체류 + 쿨다운 지났으면 Gemini 판단
            if (dwell >= DWELL_TH_SEC
                    and now - self._last_cd.get(tid, 0) > COOLDOWN_SEC):

                if self._ask_gemini(frame):
                    self._last_cd[tid] = now
                    return {"tid": tid, "frame": frame.copy()}
        else:
            self._dwell.pop(tid, None)

        return None

    def cleanup(self, active_ids: set[int]):
        gone = set(self._dwell.keys()) - active_ids
        for tid in gone:
            self._dwell.pop(tid, None)
