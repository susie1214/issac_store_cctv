"""
카카오톡 알림 모듈 — 나에게 보내기
====================================
도난/침입/낙상 등 감지 시 사장님 카카오톡으로 즉시 알림.
캡처 이미지도 함께 전송.
"""

from __future__ import annotations

import base64
import json
import os
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import cv2
import numpy as np
import socket

import requests
from dotenv import load_dotenv

load_dotenv()

REST_API_KEY  = os.getenv("KAKAO_REST_API_KEY", "")
REDIRECT_URI  = "http://localhost:38241/kakao/callback"
TOKEN_FILE    = Path("models/kakao_token.json")
COOLDOWN_SEC  = 30   # 동일 이벤트 재알림 억제 (초)


class KakaoNotifier:
    def __init__(self):
        self.ready = False
        self._token: str = ""
        self._last_sent: dict[str, float] = {}

        if not REST_API_KEY:
            print("[카카오] .env 에 KAKAO_REST_API_KEY 없음")
            return

        self._load_token()

    # ── 토큰 로드 ─────────────────────────────
    def _load_token(self):
        if TOKEN_FILE.exists():
            data = json.loads(TOKEN_FILE.read_text())
            self._token = data.get("access_token", "")
            if self._token:
                self.ready = True
                print("[카카오] 토큰 로드 완료 ✅")
            return
        print("[카카오] 토큰 없음 → python kakao_notify.py --auth 실행")

    def _save_token(self, data: dict):
        TOKEN_FILE.parent.mkdir(exist_ok=True)
        TOKEN_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        self._token = data["access_token"]
        self.ready = True

    # ── 토큰 갱신 ─────────────────────────────
    def refresh_token(self, refresh_token: str) -> bool:
        res = requests.post("https://kauth.kakao.com/oauth/token", data={
            "grant_type":    "refresh_token",
            "client_id":     REST_API_KEY,
            "client_secret": os.getenv("KAKAO_CLIENT_SECRET", ""),
            "refresh_token": refresh_token,
        })
        if res.status_code == 200:
            data = res.json()
            saved = json.loads(TOKEN_FILE.read_text())
            saved["access_token"] = data["access_token"]
            if "refresh_token" in data:
                saved["refresh_token"] = data["refresh_token"]
            self._save_token(saved)
            print("[카카오] 토큰 갱신 완료")
            return True
        return False

    # ── 나에게 보내기 ─────────────────────────
    def send(self, event_type: str, detail: str,
             frame: np.ndarray | None = None) -> bool:
        if not self.ready:
            return False

        # 쿨다운 체크
        now = time.time()
        if now - self._last_sent.get(event_type, 0) < COOLDOWN_SEC:
            return False
        self._last_sent[event_type] = now

        # 이미지 캡처 → static 폴더에 저장 (웹서버로 접근 가능)
        img_url = None
        if frame is not None:
            cap_path = Path("static/alert_capture.jpg")
            cap_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(cap_path), frame)
            # 내부 IP로 접근 가능한 URL
            try:
                import socket
                ip = socket.gethostbyname(socket.gethostname())
            except Exception:
                ip = "172.0.0.1"
            img_url = f"http://{ip}:38241/static/alert_capture.jpg"

        # 메시지 아이콘
        icons = {
            "도난의심": "🚨", "낙상감지": "🚑", "침입감지": "⚠️",
            "화재감지": "🔥", "연기감지": "💨", "소리감지": "🔊",
        }
        icon = icons.get(event_type, "ℹ️")
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = "172.30.1.34"
        dashboard_url = f"http://{ip}:38241"

        # 이미지 있으면 feed 타입, 없으면 text 타입
        if img_url:
            template = {
                "object_type": "feed",
                "content": {
                    "title": f"{icon} [{event_type}] 스마트 매장 알림",
                    "description": (
                        f"내용: {detail}\n"
                        f"시각: {now_str}"
                    ),
                    "image_url": img_url,
                    "image_width": 640,
                    "image_height": 480,
                    "link": {
                        "web_url": dashboard_url,
                        "mobile_web_url": dashboard_url,
                    },
                },
                "buttons": [{
                    "title": "📱 대시보드 확인",
                    "link": {
                        "web_url": dashboard_url,
                        "mobile_web_url": dashboard_url,
                    },
                }],
            }
        else:
            template = {
                "object_type": "text",
                "text": (
                    f"{icon} [{event_type}] 스마트 매장 알림\n\n"
                    f"내용: {detail}\n"
                    f"시각: {now_str}\n\n"
                    f"📱 대시보드 → {dashboard_url}"
                ),
                "link": {
                    "web_url": dashboard_url,
                    "mobile_web_url": dashboard_url,
                },
            }

        res = requests.post(
            "https://kapi.kakao.com/v2/api/talk/memo/default/send",
            headers={"Authorization": f"Bearer {self._token}"},
            data={"template_object": json.dumps(template, ensure_ascii=False)},
        )

        if res.status_code == 200 and res.json().get("result_code") == 0:
            print(f"[카카오] 전송 완료: {event_type}")
            return True

        # 토큰 만료 시 갱신 후 재시도
        if res.status_code == 401:
            saved = json.loads(TOKEN_FILE.read_text())
            if self.refresh_token(saved.get("refresh_token", "")):
                return self.send(event_type, detail, frame)

        print(f"[카카오] 전송 실패: {res.status_code} {res.text}")
        return False


# ── OAuth 인증 (최초 1회) ──────────────────────
def _auth_flow():
    """브라우저로 카카오 로그인 → 인가코드 수신 → 토큰 발급"""
    auth_url = (
        "https://kauth.kakao.com/oauth/authorize?"
        + urlencode({
            "client_id":     REST_API_KEY,
            "redirect_uri":  REDIRECT_URI,
            "response_type": "code",
            "scope":         "talk_message",
        })
    )
    print(f"\n브라우저가 열립니다. 카카오 로그인 후 승인해주세요.")
    webbrowser.open(auth_url)

    # 임시 HTTP 서버로 인가코드 수신
    code_holder = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            qs = parse_qs(urlparse(self.path).query)
            code_holder["code"] = qs.get("code", [""])[0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write("<h2>인증 완료! 창을 닫으세요.</h2>".encode("utf-8"))
        def log_message(self, *a): pass

    port = int(REDIRECT_URI.split(":")[-1].split("/")[0])
    srv = HTTPServer(("", port), Handler)
    srv.timeout = 60
    srv.handle_request()

    code = code_holder.get("code", "")
    if not code:
        print("인가코드 수신 실패")
        return

    # 토큰 교환
    res = requests.post("https://kauth.kakao.com/oauth/token", data={
        "grant_type":    "authorization_code",
        "client_id":     REST_API_KEY,
        "client_secret": os.getenv("KAKAO_CLIENT_SECRET", ""),
        "redirect_uri":  REDIRECT_URI,
        "code":          code,
    })
    if res.status_code == 200:
        data = res.json()
        TOKEN_FILE.parent.mkdir(exist_ok=True)
        TOKEN_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        print(f"\n✅ 카카오 토큰 발급 완료!")
        print(f"   저장 위치: {TOKEN_FILE}")
        # 테스트 메시지
        n = KakaoNotifier()
        n.send("테스트", "카카오톡 알림 연결 완료!")
    else:
        print(f"토큰 발급 실패: {res.text}")


if __name__ == "__main__":
    import sys
    if "--auth" in sys.argv:
        _auth_flow()
    else:
        print("사용법: python kakao_notify.py --auth")
