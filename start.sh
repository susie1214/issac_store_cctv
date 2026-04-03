#!/bin/bash
# OrangePi / Linux 실행 스크립트
# 사용법:
#   chmod +x start.sh
#   ./start.sh

set -e
cd "$(dirname "$0")"

# ── venv 확인 ──────────────────────────────────
if [ ! -f "venv/bin/activate" ]; then
    echo "[설치] 가상환경 생성 중..."
    python3 -m venv venv
    source venv/bin/activate
    echo "[설치] 패키지 설치 중..."
    pip install -r requirements_orangepi.txt
else
    source venv/bin/activate
fi

# ── PWA 아이콘 (최초 1회) ──────────────────────
if [ ! -f "static/icons/icon-192.png" ]; then
    python create_icons.py 2>/dev/null || true
fi

# ── 로컬 IP 출력 ───────────────────────────────
IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")
PORT=38241

echo ""
echo "======================================================"
echo "  스마트 매장 관리 시스템 (OrangePi)"
echo "======================================================"
echo "  로컬    : http://localhost:${PORT}"
echo "  스마트폰 : http://${IP}:${PORT}"
echo "  설정    : http://${IP}:${PORT}/settings"
echo "  종료    : Ctrl+C"
echo "======================================================"
echo ""

# ── OrangePi 권장 설정으로 실행 ───────────────
# 해상도 640x480, 감지해상도 100%, 프레임간격 7
python web_app.py "$@"
