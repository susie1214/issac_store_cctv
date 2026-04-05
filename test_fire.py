"""
화재/연기 감지 테스트 — 노트북 카메라
======================================
핵심: 불꽃 = 주황색 + 움직임(깜빡임) 동시 조건
      배경 주황색 = 주황색만 있고 움직임 없음 → 무시

실행: venv\Scripts\python.exe test_fire.py

조작:
  Q : 종료
  S : 더 민감하게 (임계값 낮춤)
  D : 덜 민감하게 (임계값 높임)
"""

import cv2
import numpy as np

fire_th    = 0.012   # 화재 픽셀 비율
confirm_f  = 10      # 연속 N프레임 확인
ceil_ratio = 0.25    # 상단 N% 제외

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fire_consec = 0
prev_gray   = None

print("화재 감지 테스트 (색상 + 움직임 필터)")
print("  S=민감  D=둔감  Q=종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    roi_top = int(h * ceil_ratio)
    roi = frame[roi_top:, :]

    # ── 1) 화재 색상 마스크 ──
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    fire_lo1 = np.array([0,  120, 150])
    fire_hi1 = np.array([22, 255, 255])
    fire_lo2 = np.array([158, 120, 150])
    fire_hi2 = np.array([180, 255, 255])
    color_mask = (cv2.inRange(hsv, fire_lo1, fire_hi1) |
                  cv2.inRange(hsv, fire_lo2, fire_hi2))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN,
                                   np.ones((5, 5), np.uint8))

    # ── 2) 움직임 마스크 (프레임 차분) ──
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    motion_mask = np.zeros_like(color_mask)
    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray_roi)
        _, motion_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN,
                                        np.ones((3, 3), np.uint8))
    prev_gray = gray_roi.copy()

    # ── 3) 색상 AND 움직임 = 불꽃 ──
    fire_mask = cv2.bitwise_and(color_mask, motion_mask)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_DILATE,
                                  np.ones((7, 7), np.uint8))

    # 형태 검증
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    valid_mask = np.zeros_like(fire_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) < 400:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if (bh / (bw + 1e-6)) < 0.25:
            continue
        cv2.drawContours(valid_mask, [cnt], -1, 255, -1)
        cv2.rectangle(frame, (x, y + roi_top),
                      (x + bw, y + bh + roi_top), (0, 80, 255), 2)

    fire_ratio  = cv2.countNonZero(valid_mask) / max(valid_mask.size, 1)
    fire_now    = fire_ratio > fire_th
    fire_consec = fire_consec + 1 if fire_now else 0
    fire_alert  = fire_consec >= confirm_f

    # 오버레이
    overlay = np.zeros_like(frame)
    overlay[roi_top:, :][valid_mask > 0] = (0, 60, 255)
    frame = cv2.addWeighted(frame, 1.0, overlay, 0.4, 0)

    cv2.rectangle(frame, (0, 0), (w, roi_top), (40, 40, 40), -1)
    cv2.putText(frame, "CEILING EXCLUDED", (8, roi_top - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    bar_color = (0, 0, 220) if fire_alert else (0, 160, 255) if fire_now else (60, 200, 60)
    status = "FIRE DETECTED!!!" if fire_alert else f"detecting {fire_consec}/{confirm_f}" if fire_now else "Normal"
    cv2.rectangle(frame, (0, h - 60), (w, h), (25, 25, 25), -1)
    cv2.putText(frame, f"Status: {status}", (8, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, bar_color, 2)
    cv2.putText(frame, f"Fire: {fire_ratio*100:.2f}%  TH={fire_th*100:.1f}%  S=민감 D=둔감 Q=종료",
                (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    if fire_alert:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        cv2.putText(frame, "FIRE DETECTED", (w // 2 - 130, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

    cv2.imshow("Fire Detection Test", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fire_th = max(0.001, fire_th - 0.002)
        print(f"임계값: {fire_th*100:.1f}%")
    elif key == ord('d'):
        fire_th = min(0.10, fire_th + 0.002)
        print(f"임계값: {fire_th*100:.1f}%")

cap.release()
cv2.destroyAllWindows()
