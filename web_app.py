"""
스마트 매장 방문객 관리 - 웹/앱 버전 v2.2
==========================================
FastAPI + MJPEG 스트리밍 + WebSocket 실시간 통계
모바일(PWA) + PC 겸용 | 설정 파일(config.json) 지원

실행:
  python web_app.py                          # 기본 (카메라 0번)
  python web_app.py --source 1               # 다른 카메라
  python web_app.py --source rtsp://...      # IP 카메라

접속:
  http://localhost:8000        (PC)
  http://[내IP]:8000           (같은 WiFi 스마트폰)
  http://localhost:8000/settings  (설정 화면)

최적화:
  - 매 N 프레임마다 추론 (infer_every, 기본 5)
  - 사람 감지 시 자동으로 N/2 프레임마다 (적응형)
  - 감지 해상도 축소 (detect_scale, 기본 0.5 = 절반)
  - 오렌지파이: width=640, height=480, infer_every=7 권장
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import queue
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from visitor_manager import (
    VisitorManager, detect_fire_smoke,
    generate_report, log_event, speak,
    _tts_worker, _tts_q, LOG_DIR,
    load_zone, save_zone, _event_log,
)
from sound_detector import SoundDetector

# ──────────────────────────────────────────────
#  설정 파일 관리
# ──────────────────────────────────────────────
CONFIG_FILE = Path("config.json")

DEFAULT_CONFIG: dict = {
    "camera": {
        "source": "0",
        "width":  1280,
        "height": 720,
    },
    "detection": {
        "model":        "yolov8n-pose.pt",
        "conf":         0.40,
        "infer_every":  5,      # N 프레임마다 YOLO 추론
        "detect_scale": 0.5,    # YOLO 입력 해상도 배율
    },
    "boundary": {
        "line_ratio": 0.20,     # 수직 경계선 위치 (0~1)
        "flip":       False,    # True = 우→좌가 입장
    },
}


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            merged = copy.deepcopy(DEFAULT_CONFIG)
            for k, v in data.items():
                if isinstance(v, dict) and k in merged:
                    merged[k].update(v)
                else:
                    merged[k] = v
            return merged
        except Exception as e:
            print(f"[설정] config.json 읽기 오류: {e} → 기본값 사용")
    return copy.deepcopy(DEFAULT_CONFIG)


def save_config(cfg: dict):
    CONFIG_FILE.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


cfg = load_config()


# ──────────────────────────────────────────────
#  MobileNet 나이/성별 (OpenCV DNN + Caffe)
#  모델 다운로드: python download_models.py
# ──────────────────────────────────────────────
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

AGE_LABELS    = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                 '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LABELS = ['남성', '여성']
_AG_REQUIRED  = ["age_deploy.prototxt", "age_net.caffemodel",
                 "gender_deploy.prototxt", "gender_net.caffemodel"]


class AgeGenderNet:
    """
    얼굴 감지: DNN(opencv_face_detector_uint8.pb) 우선,
               없으면 Haar Cascade(OpenCV 기본 내장) 폴백
    나이/성별: Caffe 모델 (Levi & Hassner)
    """
    FACE_CONF  = 0.70
    HAAR_SCALE = 1.1
    HAAR_MIN   = 5

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.ready       = False
        self.use_dnn_face = False

        missing = [f for f in _AG_REQUIRED if not (model_dir / f).exists()]
        if missing:
            print(f"[나이/성별] 모델 없음: {missing}")
            print("  → python download_models.py  실행 후 재시작")
            return
        try:
            self.age_net = cv2.dnn.readNet(
                str(model_dir / "age_net.caffemodel"),
                str(model_dir / "age_deploy.prototxt"))
            self.gender_net = cv2.dnn.readNet(
                str(model_dir / "gender_net.caffemodel"),
                str(model_dir / "gender_deploy.prototxt"))
        except Exception as e:
            print(f"[나이/성별] Caffe 모델 로드 실패: {e}")
            return

        pb  = model_dir / "opencv_face_detector_uint8.pb"
        pbt = model_dir / "opencv_face_detector.pbtxt"
        if pb.exists() and pbt.exists():
            try:
                self.face_net     = cv2.dnn.readNet(str(pb), str(pbt))
                self.use_dnn_face = True
                print("[나이/성별] DNN 얼굴감지 + Caffe Age/Gender ✅")
            except Exception:
                pass
        if not self.use_dnn_face:
            xml_candidates = [
                r"C:/tmp/haarcascade_frontalface_default.xml",
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
            ]
            xml = next((x for x in xml_candidates if Path(x).exists()), "")
            self.haar = cv2.CascadeClassifier(xml)
            if self.haar.empty():
                print("[나이/성별] Haar Cascade 로드 실패")
                return
            print("[나이/성별] Haar Cascade 얼굴감지 + Caffe Age/Gender ✅")

        self.ready = True

    def _face_boxes(self, frame: np.ndarray) -> list[tuple]:
        h, w = frame.shape[:2]
        if self.use_dnn_face:
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                         [104, 117, 123], True, False)
            self.face_net.setInput(blob)
            dets = self.face_net.forward()
            boxes = []
            for d in dets[0, 0]:
                if float(d[2]) < self.FACE_CONF:
                    continue
                x1 = max(0, int(d[3] * w) - 20)
                y1 = max(0, int(d[4] * h) - 20)
                x2 = min(w, int(d[5] * w) + 20)
                y2 = min(h, int(d[6] * h) + 20)
                boxes.append((x1, y1, x2, y2))
            return boxes
        else:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(
                gray, self.HAAR_SCALE, self.HAAR_MIN, minSize=(60, 60))
            return [(x, y, x + w2, y + h2)
                    for (x, y, w2, h2) in (faces if len(faces) else [])]

    def detect(self, frame: np.ndarray) -> list[dict]:
        if not self.ready or frame is None or frame.size == 0:
            return []
        results = []
        for (x1, y1, x2, y2) in self._face_boxes(frame):
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
            fblob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                [78.4263377603, 87.7689143744, 114.895847746],
                swapRB=False)
            self.gender_net.setInput(fblob)
            gender = GENDER_LABELS[self.gender_net.forward()[0].argmax()]
            self.age_net.setInput(fblob)
            age = AGE_LABELS[self.age_net.forward()[0].argmax()]
            results.append({"gender": gender, "age": age,
                             "box": (x1, y1, x2, y2)})
        return results


_age_gender_net = AgeGenderNet()
GENDER_OK       = _age_gender_net.ready
_gender_cache: dict[int, dict] = {}


# ──────────────────────────────────────────────
#  SmartCamera: 프레임 스킵 기반 YOLO 최적화
# ──────────────────────────────────────────────
class SmartCamera:
    """
    캡처 스레드(30fps)와 YOLO 스레드(N프레임마다)를 분리.

    ▸ infer_every = N  : 매 N 프레임마다 추론
    ▸ 사람 감지 시     : 자동으로 N/2 프레임마다 (적응형)
    ▸ detect_scale     : YOLO 입력 해상도 배율 (0.5 = 절반, ~4배 빠름)

    오렌지파이5 권장: width=640, height=480, detect_scale=1.0, infer_every=7
    """

    def __init__(self):
        c = cfg["camera"]
        d = cfg["detection"]
        b = cfg["boundary"]

        src = c["source"]
        self._source = int(src) if str(src).isdigit() else src

        self.vm = VisitorManager(
            model_path=d["model"],
            line_ratio=b["line_ratio"],
            flip=b["flip"],
            conf=d["conf"],
            zone_pts=load_zone(),
        )

        self._raw_frame: np.ndarray | None = None
        self._drawn_frame: np.ndarray | None = None
        self._frame_id:  int = 0
        self._lock       = threading.Lock()
        self._running    = True

        self.stats: dict = {
            "current": 0, "total": 0, "fire": 0,
            "intrusion": 0, "theft": 0, "fall": 0, "sound": 0,
            "time": "", "alerts": [],
            "gender": {}, "events": [], "gender_ok": GENDER_OK,
        }
        self._gender_count: dict[str, int] = defaultdict(int)
        self._sound_count: int = 0

        # 소리 감지
        self._sound_detector = SoundDetector(
            MODEL_DIR / "yamnet.tflite",
            callback=self._on_sound,
        )

        self._cap_thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._yolo_thread = threading.Thread(target=self._yolo_loop,    daemon=True)

    def start(self):
        self._cap_thread.start()
        self._yolo_thread.start()
        self._sound_detector.start()

    def stop(self):
        self._running = False
        self._sound_detector.stop()

    def _on_sound(self, label_kr: str, score: float):
        """소리 감지 콜백 — SoundDetector 스레드에서 호출됨"""
        self._sound_count += 1
        msg = f"소리 감지: {label_kr} ({score:.0%})"
        self.vm._add_alert(msg, (0, 180, 255))
        speak(f"{label_kr}이 감지되었습니다")
        log_event("소리감지", f"{label_kr} score={score:.2f}")

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            frame = self._drawn_frame
        if frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        return buf.tobytes()

    def get_raw_jpeg(self) -> bytes | None:
        """설정 페이지 스냅샷 — 오버레이 없는 원본"""
        with self._lock:
            frame = self._raw_frame
        if frame is None:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes()

    # ── 캡처 스레드 (30fps) ───────────────────
    def _capture_loop(self):
        import os
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
        cap = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[오류] 카메라 열기 실패: {self._source}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg["camera"]["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
        cap.set(cv2.CAP_PROP_FPS, 30)

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            with self._lock:
                self._raw_frame = frame.copy()
                self._frame_id += 1
        cap.release()

    # ── YOLO 스레드 (N 프레임마다) ────────────
    def _yolo_loop(self):
        last_id = -1
        while self._running:
            # ── 적응형 프레임 스킵 ──
            n = cfg["detection"]["infer_every"]
            if self.vm.current_people > 0:
                n = max(1, n // 2)   # 사람 있으면 2배 빠르게

            with self._lock:
                fid   = self._frame_id
                frame = self._raw_frame

            if frame is None or (fid - last_id) < n:
                time.sleep(0.005)
                continue

            last_id = fid

            # ── 해상도 축소 → YOLO 추론 ──
            scale = cfg["detection"]["detect_scale"]
            small = (cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                     if scale < 1.0 else frame)

            processed = self.vm.process(small)

            if scale < 1.0:
                h, w = frame.shape[:2]
                processed = cv2.resize(processed, (w, h))

            self._update_gender(frame)
            self._update_stats()

            with self._lock:
                self._drawn_frame = processed

    def _update_gender(self, frame: np.ndarray):
        if not GENDER_OK:
            return
        new_ids = set(self.vm.prev_x.keys()) - set(_gender_cache.keys())
        if not new_ids:
            return
        faces = _age_gender_net.detect(frame)
        for f in faces:
            fx = (f["box"][0] + f["box"][2]) / 2
            best = min(new_ids,
                       key=lambda tid: abs(self.vm.prev_x.get(tid, 0) - fx),
                       default=None)
            if best is None:
                continue
            _gender_cache[best] = {"gender": f["gender"], "age": f["age"]}
            self._gender_count[f["gender"]] = \
                self._gender_count.get(f["gender"], 0) + 1

    def _update_stats(self):
        vm     = self.vm
        alerts = [
            {"text": t,
             "color": "#{:02x}{:02x}{:02x}".format(int(c[2]), int(c[1]), int(c[0]))}
            for t, c, tm in vm._alerts if tm > 0
        ]
        events = [{"time": e["time"], "type": e["type"], "detail": e["detail"]}
                  for e in list(_event_log)[-15:]]
        self.stats = {
            "current":   vm.current_people,
            "total":     vm.total_visitors,
            "fire":      vm.fire_count,
            "intrusion": vm.intrusion_count,
            "theft":     vm.theft_count,
            "fall":      vm.fall_count,
            "sound":     self._sound_count,
            "time":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "alerts":    alerts,
            "events":    events,
            "gender":    dict(self._gender_count),
            "gender_ok": GENDER_OK,
        }


# ──────────────────────────────────────────────
#  FastAPI
# ──────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="스마트 매장 관리")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

cam: SmartCamera | None = None
_ws_clients: list[WebSocket] = []


@app.on_event("startup")
async def startup():
    global cam
    cam = SmartCamera()
    cam.start()

    t = threading.Thread(target=_tts_worker, daemon=True)
    t.start()

    asyncio.create_task(_ws_broadcast())


async def _ws_broadcast():
    while True:
        await asyncio.sleep(0.5)
        if not _ws_clients or cam is None:
            continue
        data = json.dumps(cam.stats, ensure_ascii=False)
        dead = []
        for ws in _ws_clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _ws_clients.remove(ws)


# ── 비디오 스트림 ─────────────────────────────
@app.get("/video")
def video_feed():
    def gen():
        while True:
            if cam is None:
                time.sleep(0.05)
                continue
            jpeg = cam.get_jpeg()
            if jpeg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpeg + b"\r\n")
            time.sleep(0.033)
    return StreamingResponse(gen(),
        media_type="multipart/x-mixed-replace; boundary=frame")


# ── WebSocket ─────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ── 보고서 ───────────────────────────────────
@app.get("/api/report")
async def api_report():
    if cam is None:
        return JSONResponse({"error": "카메라 미연결"})
    vm  = cam.vm
    rpt = generate_report(vm.total_visitors, vm.fire_count, vm.intrusion_count)
    return JSONResponse({"path": str(rpt), "ok": True})


# ── 설정 API ─────────────────────────────────
@app.get("/api/config")
async def get_config():
    return JSONResponse(cfg)


@app.post("/api/config")
async def post_config(request: Request):
    data = await request.json()
    for section, values in data.items():
        if section in cfg and isinstance(values, dict):
            cfg[section].update(values)
        else:
            cfg[section] = values
    save_config(cfg)
    # 즉시 적용 가능한 설정
    if cam:
        cam.vm.line_ratio = cfg["boundary"]["line_ratio"]
        cam.vm.flip       = cfg["boundary"]["flip"]
        cam.vm.conf       = cfg["detection"]["conf"]
    return JSONResponse({"ok": True, "config": cfg})


@app.get("/api/snapshot")
async def snapshot():
    """설정 페이지용 현재 프레임 (오버레이 없음)"""
    if cam is None:
        return Response(status_code=503)
    data = cam.get_raw_jpeg()
    if data is None:
        return Response(status_code=503)
    return Response(data, media_type="image/jpeg")


@app.post("/api/zone")
async def api_zone(request: Request):
    data = await request.json()
    pts  = data.get("points", [])
    if cam:
        cam.vm.zone_pts = pts
    save_zone(pts)
    return JSONResponse({"ok": True})


# ── PWA 필수 파일 ─────────────────────────────
@app.get("/manifest.json")
async def pwa_manifest():
    f = STATIC_DIR / "manifest.json"
    return Response(f.read_bytes(), media_type="application/manifest+json")

@app.get("/sw.js")
async def pwa_sw():
    f = STATIC_DIR / "sw.js"
    return Response(f.read_bytes(), media_type="application/javascript")

# ── HTML 페이지 (templates/ 폴더에서 로드) ──
TEMPLATE_DIR = Path(__file__).parent / "templates"

@app.get("/", response_class=HTMLResponse)
async def index():
    return (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")

@app.get("/settings", response_class=HTMLResponse)
async def settings():
    return (TEMPLATE_DIR / "settings.html").read_text(encoding="utf-8")






# ──────────────────────────────────────────────
#  실행
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="스마트 매장 웹 서버")
    p.add_argument("--source", help="카메라 소스 (설정 파일 우선 적용 후 덮어씀)")
    p.add_argument("--port",   type=int, default=38241)
    return p.parse_args()


if __name__ == "__main__":
    import socket

    args = parse_args()
    if args.source is not None:
        cfg["camera"]["source"] = args.source

    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]; s.close(); return ip
        except Exception:
            return "localhost"

    ip = get_local_ip()
    ie = cfg["detection"]["infer_every"]
    sc = cfg["detection"]["detect_scale"]
    w  = cfg["camera"]["width"]
    h  = cfg["camera"]["height"]

    print("\n" + "=" * 58)
    print("  🏪 스마트 매장 방문객 관리 시스템  웹/앱 v2.2")
    print("=" * 58)
    print(f"  PC 웹    : http://localhost:{args.port}")
    print(f"  스마트폰 : http://{ip}:{args.port}  (같은 WiFi)")
    print(f"  설정화면 : http://localhost:{args.port}/settings")
    print()
    import re
    _src_display = re.sub(r'(:)([^@]+)(@)', r'\1****\3', cfg['camera']['source'])
    print(f"  [카메라]  소스={_src_display}  {w}×{h}")
    print(f"  [추론]    매 {ie}프레임마다  감지해상도={int(sc*100)}%")
    print(f"            (사람 감지 시 {max(1,ie//2)}프레임마다 자동 전환)")
    print(f"  [모델]    {cfg['detection']['model']}")
    print()
    print(f"  [나이/성별] {'✅ MobileNet 활성' if GENDER_OK else '❌ 비활성 (python download_models.py)'}")
    print()
    print("  [오렌지파이5 권장 설정]  /settings 에서 변경")
    print("    해상도 640×480 | 감지해상도 100% | 프레임간격 7")
    print("=" * 58 + "\n")

    import signal, sys
    def _shutdown(sig, frame):
        print("\n[종료] 서버를 종료합니다...")
        if cam:
            cam.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")