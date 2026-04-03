"""
낙상/쓰러짐 감지 모듈 v2 — YOLO Pose keypoint 기반
====================================================
yolov8n-pose.pt 의 17개 관절 좌표로 상체 기울기(척추 각도)를 계산.
keypoint 신뢰도가 낮으면 bbox 비율로 자동 폴백.

COCO keypoint 인덱스:
  0:코  1:왼눈  2:오른눈  3:왼귀  4:오른귀
  5:왼어깨  6:오른어깨  7:왼팔꿈치  8:오른팔꿈치
  9:왼손목  10:오른손목
  11:왼엉덩이  12:오른엉덩이
  13:왼무릎  14:오른무릎  15:왼발목  16:오른발목

판정 기준:
  어깨중점 → 엉덩이중점 벡터가 수직에서 FALL_ANGLE_TH° 이상 기울면 낙상 후보
  → FALL_CONFIRM_F 프레임 연속 확인 후 확정 (오탐 방지)
"""

from __future__ import annotations

import math

# ── 파라미터 ──────────────────────────────────────
FALL_ANGLE_TH   = 55.0   # 척추 수직 기울기 임계값 (도). 이상이면 쓰러진 것으로 간주
FALL_CONFIRM_F  = 8      # 연속 N 프레임 이상이어야 낙상 확정
FALL_COOLDOWN_F = 200    # 동일 ID 재경보 억제 프레임
FALL_MIN_H      = 50     # bbox 높이 최솟값 (px) — 너무 멀리 있는 사람 무시
KP_CONF_TH      = 0.30   # keypoint 신뢰도 임계값

# COCO keypoint 인덱스
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_HIP,      _R_HIP      = 11, 12


class FallDetector:
    """
    VisitorManager.process() 에서 per-frame 호출.

    사용법:
        fd = FallDetector()

        # 매 프레임, 각 사람마다
        new_fall = fd.update(tid, kps, bw, bh)

        # 프레임 끝에 한 번
        fd.cleanup(current_ids)
    """

    def __init__(
        self,
        angle_th: float     = FALL_ANGLE_TH,
        confirm_frames: int  = FALL_CONFIRM_F,
        cooldown: int        = FALL_COOLDOWN_F,
        min_height: int      = FALL_MIN_H,
    ):
        self.angle_th       = angle_th
        self.confirm_frames = confirm_frames
        self.cooldown       = cooldown
        self.min_height     = min_height

        self._consec: dict[int, int] = {}   # tid → 연속 낙상 프레임 수
        self._fallen: set[int]       = set() # 현재 낙상 상태 ID
        self._cd: dict[int, int]     = {}   # tid → 남은 쿨다운

    # ── keypoint 기반 척추 기울기 계산 ───────────
    def _torso_angle(self, kps) -> float | None:
        """
        kps : np.ndarray shape (17, 3) — (x, y, confidence)
        반환 : 척추가 수직에서 기운 각도 (0°=직립, 90°=수평)
               keypoint 신뢰도 부족 시 None
        """
        def midpoint(i, j):
            ci, cj = float(kps[i][2]), float(kps[j][2])
            if ci >= KP_CONF_TH and cj >= KP_CONF_TH:
                return ((kps[i][0] + kps[j][0]) / 2,
                        (kps[i][1] + kps[j][1]) / 2)
            if ci >= KP_CONF_TH:
                return float(kps[i][0]), float(kps[i][1])
            if cj >= KP_CONF_TH:
                return float(kps[j][0]), float(kps[j][1])
            return None

        shoulder = midpoint(_L_SHOULDER, _R_SHOULDER)
        hip      = midpoint(_L_HIP,      _R_HIP)

        if shoulder is None or hip is None:
            return None

        dx = shoulder[0] - hip[0]
        dy = shoulder[1] - hip[1]  # 서 있을 때 음수 (어깨가 위)

        # 수직 기준 각도: 0° = 완전 직립, 90° = 완전 수평
        angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
        return angle

    # ── bbox 비율 폴백 ────────────────────────────
    def _bbox_fallen(self, bw: float, bh: float) -> bool:
        """keypoint 없을 때 bbox 가로/세로 비율로 낙상 판단"""
        return bh >= self.min_height and (bw / bh) >= 1.35

    # ── 핵심: 매 프레임 각 사람마다 호출 ──────────
    def update(self, tid: int, kps, bw: float, bh: float) -> bool:
        """
        tid : ByteTracker track ID
        kps : np.ndarray (17, 3) 또는 None (pose 모델 아닐 때)
        bw  : bbox 너비
        bh  : bbox 높이

        Returns True → 이번에 새로 낙상 확정 (경보 발령 타이밍)
        """
        # 쿨다운 감소만 하고 종료
        if self._cd.get(tid, 0) > 0:
            self._cd[tid] -= 1
            return False

        # 너무 작은 사람 무시
        if bh < self.min_height:
            return False

        # 낙상 여부 판단 (keypoint 우선, 폴백은 bbox 비율)
        if kps is not None:
            angle = self._torso_angle(kps)
            fallen_candidate = (
                angle > self.angle_th if angle is not None
                else self._bbox_fallen(bw, bh)   # keypoint 부족 → 폴백
            )
        else:
            fallen_candidate = self._bbox_fallen(bw, bh)

        if fallen_candidate:
            self._consec[tid] = self._consec.get(tid, 0) + 1
        else:
            # 정상 자세 복귀
            self._consec[tid] = 0
            self._fallen.discard(tid)
            return False

        # 연속 N 프레임 이상 & 첫 확정
        if (self._consec[tid] >= self.confirm_frames
                and tid not in self._fallen):
            self._fallen.add(tid)
            self._cd[tid] = self.cooldown
            return True

        return False

    def is_fallen(self, tid: int) -> bool:
        return tid in self._fallen

    def cleanup(self, active_ids: set[int]):
        """사라진 ID 정리"""
        gone = set(self._consec.keys()) - active_ids
        for tid in gone:
            self._consec.pop(tid, None)
            self._fallen.discard(tid)
            self._cd.pop(tid, None)
