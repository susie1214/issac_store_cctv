"""
지능형 매장 방문객 관리 시스템 v2.0
=====================================
기능:
  - 수직 경계선 출입 감지 (좌=출입문, 좌→우=입장)
  - 침입 금지 구역 감지 (우측 구역)
  - 안정화된 카운팅 (동일인 중복 방지)
  - 화재 / 연기 감지
  - 일별 매장 보고서 자동 생성

조작키:
  Q      : 종료 + 보고서 생성
  U / D  : 경계선 좌/우 이동
  Z      : 침입구역 마우스 설정 모드 (Z 누르고 클릭 → Enter 확정 → Z 재입력 취소)
  R      : 지금 즉시 보고서 생성
  F      : 방향 반전

좌표 설정 방법:
  1. 실행 후 Z 키를 누릅니다.
  2. 마우스로 침입 금지 구역의 꼭짓점을 순서대로 클릭합니다.
  3. Enter 키를 누르면 구역이 확정됩니다.
  4. 설정값은 zone_config.json 에 자동 저장됩니다.
  → 다음 실행 시 자동 로드합니다.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import queue
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from fall_detector import FallDetector
from face_manager import FaceManager
from theft_detector import TheftDetector
from kakao_notify import KakaoNotifier

# ── 헤드리스 감지 (OrangePi/SSH 환경 = 디스플레이 없음) ──
HEADLESS = (
    not os.environ.get("DISPLAY", "")          # X11 없음
    and not os.environ.get("WAYLAND_DISPLAY", "") # Wayland 없음
    and sys.platform != "win32"                # Windows 제외
)
if HEADLESS:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # Qt 경고 억제

# ──────────────────────────────────────────────
#  로거
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VM")

# ──────────────────────────────────────────────
#  한글 폰트 (Pillow)
# ──────────────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont
    _FONT_CANDIDATES = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    _FP = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
    if _FP:
        F_LG = ImageFont.truetype(_FP, 34)
        F_MD = ImageFont.truetype(_FP, 22)
        F_SM = ImageFont.truetype(_FP, 15)
        USE_PIL = True
    else:
        USE_PIL = False
except ImportError:
    USE_PIL = False


def kr(frame, text, pos, font=None, color=(255, 255, 255)):
    if not USE_PIL or font is None:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, color, 2, cv2.LINE_AA)
        return frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img).text(pos, text, font=font,
                             fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ──────────────────────────────────────────────
#  오디오 파일 재생 (m4a/mp3/wav)
# ──────────────────────────────────────────────
_AUDIO_DIR = Path(__file__).parent
_ENTER_SOUND = _AUDIO_DIR / "어서오세요.m4a"
_EXIT_SOUND  = _AUDIO_DIR / "안녕히가세요.m4a"

def _play_audio_async(path: Path):
    """오디오 파일 재생 — pygame(Linux/OrangePi) 우선, Windows는 PowerShell"""
    def _play():
        try:
            if sys.platform == "win32":
                import subprocess
                abs_path = str(path.resolve())
                ps_cmd = (
                    f"Add-Type -AssemblyName presentationCore; "
                    f"$p = New-Object system.windows.media.mediaplayer; "
                    f"$p.open([uri]'{abs_path}'); "
                    f"$p.Play(); "
                    f"Start-Sleep 5; "
                    f"$p.Stop()"
                )
                subprocess.run(["powershell", "-Command", ps_cmd],
                               capture_output=True, timeout=8)
            else:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(str(path))
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
        except Exception as e:
            logger.warning(f"오디오 재생 오류: {e}")
    threading.Thread(target=_play, daemon=True).start()


def _beep_async(pattern: list[tuple[float, float]]):
    """
    크로스플랫폼 비프음 — (주파수Hz, 지속ms) 리스트
    Windows: winsound, Linux: pygame sine wave 생성
    """
    def _play():
        try:
            if sys.platform == "win32":
                import winsound
                for freq, dur in pattern:
                    winsound.Beep(int(freq), int(dur))
                    time.sleep(0.05)
            else:
                import numpy as np_
                import pygame
                pygame.mixer.init(frequency=44100, size=-16, channels=1)
                for freq, dur in pattern:
                    frames = int(44100 * dur / 1000)
                    t = np_.linspace(0, dur / 1000, frames, False)
                    wave = (np_.sin(2 * np_.pi * freq * t) * 32767).astype(np_.int16)
                    sound = pygame.sndarray.make_sound(wave)
                    sound.play()
                    pygame.time.wait(int(dur) + 50)
        except Exception as e:
            logger.warning(f"비프음 오류: {e}")
    threading.Thread(target=_play, daemon=True).start()


def play_enter():
    if _ENTER_SOUND.exists():
        _play_audio_async(_ENTER_SOUND)
    else:
        speak("어서오세요")


def play_exit():
    if _EXIT_SOUND.exists():
        _play_audio_async(_EXIT_SOUND)
    else:
        speak("안녕히 가세요")


# ──────────────────────────────────────────────
#  도난 알람
# ──────────────────────────────────────────────
_theft_alarm_active = False

def play_theft_alarm():
    """도난 알람: 긴급 경보음 (도난 특화 — 높고 짧은 비프 반복)"""
    global _theft_alarm_active
    if _theft_alarm_active:
        return
    _theft_alarm_active = True
    def _alarm():
        global _theft_alarm_active
        # 높은 주파수 교차 5회 — 귀에 잘 들리는 경보 패턴
        pattern = [(2800, 180), (2200, 180)] * 5
        _beep_async(pattern)
        time.sleep(2)
        _theft_alarm_active = False
    threading.Thread(target=_alarm, daemon=True).start()
    logger.warning("🚨 도난 알람 발동!")


def play_fire_alarm():
    """화재/연기 알람: 소방 경보 패턴 (단속음 반복)"""
    # 소방 경보 특유의 단속 패턴 (높음-짧은침묵 반복)
    pattern = [(3000, 300), (0, 100)] * 6
    _beep_async(pattern)
    logger.warning("🔥 화재 알람 발동!")


# ──────────────────────────────────────────────
#  TTS
# ──────────────────────────────────────────────
_tts_q: queue.Queue = queue.Queue()
_tts_engine = None


def _tts_worker():
    """TTS 전용 스레드.
    우선순위: ① win32com SAPI5 직접 (가장 안정적, 한국어 정확)
              ② pyttsx3 (설치된 경우)
              ③ winsound 비프음 (폴백)
    """
    # ── COM 초기화 (SAPI5 필수) ──
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except Exception:
        pass

    # ── ① win32com SAPI5 직접 사용 ──────────────
    _speaker = None
    try:
        import win32com.client
        _speaker = win32com.client.Dispatch("SAPI.SpVoice")
        voices = _speaker.GetVoices()
        kor_voice = None
        for i in range(voices.Count):
            v = voices.Item(i)
            vid = v.Id.upper()
            desc = v.GetDescription().upper()
            # 한국어 음성: ID에 0412(KOR 언어코드) 또는 이름에 HEAMI/KOREAN 포함
            if "0412" in vid or "HEAMI" in desc or "KOREAN" in desc:
                kor_voice = v
                break
        if kor_voice:
            _speaker.Voice = kor_voice
            logger.info(f"TTS: 한국어 음성 선택 → {kor_voice.GetDescription()}")
        else:
            logger.warning("TTS: 한국어 음성 없음 — Windows 설정>언어>한국어>음성 팩 설치 필요")
        _speaker.Rate   = 0    # -10(느림)~10(빠름), 0=보통
        _speaker.Volume = 100
        use_mode = "win32com"
        logger.info("TTS: win32com SAPI5 초기화 성공")
    except Exception as e:
        logger.warning(f"TTS: win32com 불가 ({e})")
        _speaker = None

        # ── ② pyttsx3 폴백 ──────────────────────
        global _tts_engine
        try:
            import pyttsx3
            _tts_engine = pyttsx3.init()
            voices = _tts_engine.getProperty('voices')
            # ID에 0412 포함 여부로 한국어 음성 검색 (name보다 정확)
            kor_voice = next(
                (v for v in voices if "0412" in v.id.upper()
                 or "heami" in v.name.lower() or "korean" in v.name.lower()),
                None
            )
            if kor_voice:
                _tts_engine.setProperty('voice', kor_voice.id)
                logger.info(f"TTS: pyttsx3 한국어 음성 → {kor_voice.name}")
            else:
                logger.warning("TTS: pyttsx3 한국어 음성 없음 → 기본 음성 사용")
            _tts_engine.setProperty('rate', 150)
            _tts_engine.setProperty('volume', 1.0)
            use_mode = "pyttsx3"
            logger.info("TTS: pyttsx3 초기화 성공")
        except Exception as e2:
            logger.warning(f"TTS: pyttsx3 불가 ({e2}) → winsound 비프음 사용")
            use_mode = "winsound"

    # ── 메시지 루프 ──────────────────────────────
    while True:
        text = _tts_q.get()
        if text is None:
            break
        try:
            if use_mode == "win32com" and _speaker:
                # SVSFlagsAsync=1 → 비동기, SVSFDefault=0 → 동기(완료까지 대기)
                _speaker.Speak(text, 0)   # 0 = 동기 (완료 후 반환)

            elif use_mode == "pyttsx3" and _tts_engine:
                _tts_engine.say(text)
                _tts_engine.runAndWait()

            else:
                import winsound
                if '어서' in text:
                    winsound.Beep(1047, 200); time.sleep(0.05)
                    winsound.Beep(1319, 300)
                elif '안녕히' in text:
                    winsound.Beep(1319, 200); time.sleep(0.05)
                    winsound.Beep(1047, 300)
                elif '침입' in text:
                    for _ in range(3):
                        winsound.Beep(2000, 150); time.sleep(0.08)
                elif '화재' in text or '연기' in text:
                    for _ in range(5):
                        winsound.Beep(2500, 200); time.sleep(0.1)
                else:
                    winsound.Beep(800, 300)

        except Exception as e:
            logger.warning(f"TTS 재생 오류: {e}")
        _tts_q.task_done()


def speak(text: str):
    if _tts_q.empty():
        _tts_q.put(text)


# ──────────────────────────────────────────────
#  로그 / 보고서
# ──────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

_event_log: list[dict] = []   # 실시간 이벤트 누적


def log_event(etype: str, detail: str = ""):
    now = datetime.now()
    row = {"time": now.strftime("%H:%M:%S"), "date": now.strftime("%Y-%m-%d"),
           "type": etype, "detail": detail}
    _event_log.append(row)
    csv_path = LOG_DIR / f"events_{now.strftime('%Y%m%d')}.csv"
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["time", "date", "type", "detail"])
        if is_new:
            w.writeheader()
        w.writerow(row)


def _linear_trend(values: list) -> list:
    """단순 선형회귀 추세선 계산"""
    n = len(values)
    if n < 2:
        return values[:]
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    denom = sum((x - mx) ** 2 for x in xs) or 1
    slope = sum((xs[i] - mx) * (values[i] - my) for i in range(n)) / denom
    intercept = my - slope * mx
    return [round(intercept + slope * x, 2) for x in xs]


def _report_html(title: str, subtitle: str, labels: list, visitor_data: list,
                 kpis: list, event_rows_html: str, donut_data: dict) -> str:
    """Chart.js 기반 공통 HTML 리포트 생성"""
    trend = _linear_trend(visitor_data)
    labels_js    = str(labels)
    visitors_js  = str(visitor_data)
    trend_js     = str(trend)
    donut_labels = str(list(donut_data.keys()))
    donut_values = str(list(donut_data.values()))
    donut_colors = str([
        "#e53e3e","#e07800","#9b2335","#6b21a8","#1a6fd4","#059669"
    ][:len(donut_data)])

    kpi_html = ""
    for num, label, color in kpis:
        kpi_html += f"""
      <div class="card">
        <div class="card-num" style="color:{color}">{num}</div>
        <div class="card-label">{label}</div>
      </div>"""

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'맑은 고딕',sans-serif;background:#f0f2f5;color:#1a1a2e;padding:20px}}
  .header{{background:linear-gradient(135deg,#1a6fd4,#0d47a1);color:#fff;
           border-radius:14px;padding:22px 28px;margin-bottom:20px}}
  .header h1{{font-size:20px;font-weight:700;margin-bottom:4px}}
  .header p{{font-size:12px;opacity:.8}}
  .cards{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}}
  .card{{background:#fff;border-radius:12px;padding:18px 22px;flex:1;min-width:130px;
         box-shadow:0 2px 12px rgba(0,0,0,.07)}}
  .card-num{{font-size:36px;font-weight:800;line-height:1}}
  .card-label{{font-size:12px;color:#888;margin-top:6px}}
  .section{{background:#fff;border-radius:12px;padding:20px;
            box-shadow:0 2px 12px rgba(0,0,0,.07);margin-bottom:20px}}
  .section h2{{font-size:14px;font-weight:700;margin-bottom:16px;
               padding-left:10px;border-left:4px solid #1a6fd4;color:#333}}
  .charts{{display:grid;grid-template-columns:2fr 1fr;gap:16px;margin-bottom:20px}}
  @media(max-width:600px){{.charts{{grid-template-columns:1fr}}}}
  .chart-box{{background:#fff;border-radius:12px;padding:20px;
              box-shadow:0 2px 12px rgba(0,0,0,.07)}}
  .chart-box h2{{font-size:14px;font-weight:700;margin-bottom:14px;
                 padding-left:10px;border-left:4px solid #1a6fd4;color:#333}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  th{{background:#1a1a2e;color:#fff;padding:10px 12px;text-align:left;font-weight:600}}
  td{{padding:8px 12px;border-bottom:1px solid #f0f0f0}}
  tr:hover td{{background:#f8faff}}
  .et-입장{{color:#1a6fd4;font-weight:700}}
  .et-퇴장{{color:#aaa}}
  .et-침입감지{{color:#e07800;font-weight:700}}
  .et-화재감지{{color:#e53e3e;font-weight:700}}
  .et-연기감지{{color:#c0392b;font-weight:700}}
  .et-도난의심{{color:#9b2335;font-weight:700}}
  .et-낙상감지{{color:#6b21a8;font-weight:700}}
  .et-소리감지{{color:#059669;font-weight:700}}
  .footer{{text-align:center;color:#bbb;font-size:11px;margin-top:16px;padding-bottom:10px}}
</style>
</head>
<body>
<div class="header">
  <h1>{title}</h1>
  <p>{subtitle} &nbsp;|&nbsp; ThingsWell AI Monitor</p>
</div>

<div class="cards">{kpi_html}
</div>

<div class="charts">
  <div class="chart-box">
    <h2>방문객 추이 &amp; 추세선</h2>
    <canvas id="lineChart" height="180"></canvas>
  </div>
  <div class="chart-box">
    <h2>이벤트 분포</h2>
    <canvas id="donutChart" height="180"></canvas>
  </div>
</div>

<div class="section">
  <h2>이벤트 로그 (최근 100건)</h2>
  <table>
    <tr><th>날짜</th><th>시간</th><th>이벤트</th><th>내용</th></tr>
    {event_rows_html}
  </table>
</div>
<div class="footer">ThingsWell Visitor Management System v2.0 &nbsp;|&nbsp; 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<script>
// ── 방문객 추이 + 추세선 ──
new Chart(document.getElementById('lineChart'), {{
  data: {{
    labels: {labels_js},
    datasets: [
      {{
        type: 'bar',
        label: '방문객',
        data: {visitors_js},
        backgroundColor: 'rgba(26,111,212,0.25)',
        borderColor: '#1a6fd4',
        borderWidth: 1.5,
        borderRadius: 4,
        order: 2,
      }},
      {{
        type: 'line',
        label: '추세선',
        data: {trend_js},
        borderColor: '#e53e3e',
        borderWidth: 2.5,
        borderDash: [6,3],
        pointRadius: 0,
        tension: 0.4,
        fill: false,
        order: 1,
      }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }},
      tooltip: {{ mode: 'index' }}
    }},
    scales: {{
      y: {{ beginAtZero: true, ticks: {{ stepSize: 1 }} }},
      x: {{ ticks: {{ font: {{ size: 10 }} }} }}
    }}
  }}
}});

// ── 이벤트 도넛 ──
new Chart(document.getElementById('donutChart'), {{
  type: 'doughnut',
  data: {{
    labels: {donut_labels},
    datasets: [{{ data: {donut_values}, backgroundColor: {donut_colors}, borderWidth: 2 }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ font: {{ size: 10 }}, padding: 8 }} }}
    }}
  }}
}});
</script>
</body></html>"""


def generate_report(total_visitors: int, fire_count: int, intrusion_count: int):
    """HTML 일별 보고서 — Chart.js 추세선 포함"""
    now  = datetime.now()
    fname = LOG_DIR / f"report_{now.strftime('%Y%m%d_%H%M%S')}.html"

    # 시간대별 집계
    hourly: dict[int, int] = defaultdict(int)
    event_types: dict[str, int] = defaultdict(int)
    for e in _event_log:
        event_types[e["type"]] += 1
        if e["type"] == "입장":
            try:
                hourly[int(e["time"].split(":")[0])] += 1
            except Exception:
                pass

    labels  = [f"{h:02d}시" for h in range(9, 22)]
    visitors = [hourly.get(h, 0) for h in range(9, 22)]

    theft_count = event_types.get("도난의심", 0)
    fall_count  = event_types.get("낙상감지", 0)

    kpis = [
        (total_visitors, "총 방문객 (명)",    "#1a6fd4"),
        (fire_count,     "화재/연기 (건)",    "#e53e3e"),
        (intrusion_count,"침입 감지 (건)",    "#e07800"),
        (theft_count,    "도난 의심 (건)",    "#9b2335"),
        (fall_count,     "낙상 감지 (건)",    "#6b21a8"),
    ]

    donut = {k: v for k, v in event_types.items() if v > 0}
    if not donut:
        donut = {"데이터 없음": 1}

    event_rows_html = "".join(
        f'<tr><td>{e.get("date","")}</td><td>{e["time"]}</td>'
        f'<td class="et-{e["type"]}">{e["type"]}</td><td>{e.get("detail","")}</td></tr>'
        for e in reversed(_event_log[-100:])
    )

    html = _report_html(
        title=f"매장 일일 보고서 — {now.strftime('%Y년 %m월 %d일')}",
        subtitle=now.strftime("%Y년 %m월 %d일"),
        labels=labels, visitor_data=visitors,
        kpis=kpis, event_rows_html=event_rows_html, donut_data=donut,
    )

    fname.write_text(html, encoding="utf-8")
    logger.info(f"보고서 저장: {fname}")
    try:
        os.startfile(str(fname))
    except Exception:
        pass
    return fname


def _load_csv_range(start_date: datetime, end_date: datetime) -> list[dict]:
    """start_date ~ end_date 범위의 모든 events_YYYYMMDD.csv 로드"""
    rows = []
    cur = start_date
    while cur <= end_date:
        csv_path = LOG_DIR / f"events_{cur.strftime('%Y%m%d')}.csv"
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                rows.extend(list(csv.DictReader(f)))
        cur += timedelta(days=1)
    return rows


def _make_period_report(rows: list[dict], title: str, subtitle: str) -> Path:
    """주간/월간 공통 HTML 보고서 — Chart.js 추세선 + 도넛 차트"""
    from collections import Counter
    now = datetime.now()
    fname = LOG_DIR / f"report_{now.strftime('%Y%m%d_%H%M%S')}.html"

    # 날짜별 입장 집계 (정렬된 날짜 순서 유지)
    daily: dict[str, int] = Counter(
        e["date"] for e in rows if e.get("type") == "입장"
    )
    sorted_dates = sorted(daily.keys())

    fire_cnt    = sum(1 for e in rows if e.get("type") == "화재감지")
    smoke_cnt   = sum(1 for e in rows if e.get("type") == "연기감지")
    intrude_cnt = sum(1 for e in rows if e.get("type") == "침입감지")
    theft_cnt   = sum(1 for e in rows if e.get("type") == "도난의심")
    fall_cnt    = sum(1 for e in rows if e.get("type") == "낙상감지")
    total       = sum(daily.values())

    # 차트 데이터: MM/DD 레이블, 날짜별 방문객 수
    labels   = [d[5:] for d in sorted_dates]   # YYYY-MM-DD → MM-DD
    visitors = [daily[d] for d in sorted_dates]
    if not labels:
        labels, visitors = ["데이터 없음"], [0]

    kpis = [
        (total,               "총 방문객 (명)",  "#1a6fd4"),
        (fire_cnt + smoke_cnt,"화재/연기 (건)",  "#e53e3e"),
        (intrude_cnt,         "침입 감지 (건)",  "#e07800"),
        (theft_cnt,           "도난 의심 (건)",  "#9b2335"),
        (fall_cnt,            "낙상 감지 (건)",  "#6b21a8"),
    ]

    # 이벤트 유형 분포 (도넛 차트용)
    event_types: dict[str, int] = defaultdict(int)
    for e in rows:
        if e.get("type"):
            event_types[e["type"]] += 1
    donut = {k: v for k, v in event_types.items() if v > 0} or {"데이터 없음": 1}

    event_rows_html = "".join(
        f'<tr><td>{e.get("date","")}</td><td>{e.get("time","")}</td>'
        f'<td class="et-{e.get("type","")}">{e.get("type","")}</td>'
        f'<td>{e.get("detail","")}</td></tr>'
        for e in rows[-100:]
    )

    html = _report_html(
        title=title, subtitle=subtitle,
        labels=labels, visitor_data=visitors,
        kpis=kpis, event_rows_html=event_rows_html, donut_data=donut,
    )

    fname.write_text(html, encoding="utf-8")
    logger.info(f"보고서 저장: {fname}")
    try:
        os.startfile(str(fname))
    except Exception:
        pass
    return fname


def generate_weekly_report() -> Path:
    """이번 주(월~오늘) 보고서"""
    now = datetime.now()
    start = now - timedelta(days=now.weekday())   # 이번 주 월요일
    start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    rows = _load_csv_range(start, now)
    title = f"매장 주간 보고서 — {start.strftime('%m/%d')} ~ {now.strftime('%m/%d')}"
    subtitle = f"{start.strftime('%Y년 %m월 %d일')} ~ {now.strftime('%Y년 %m월 %d일')}"
    return _make_period_report(rows, title, subtitle)


def generate_monthly_report() -> Path:
    """이번 달 보고서"""
    now = datetime.now()
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    rows = _load_csv_range(start, now)
    title = f"매장 월간 보고서 — {now.strftime('%Y년 %m월')}"
    subtitle = f"{now.strftime('%Y년 %m월')} 전체"
    return _make_period_report(rows, title, subtitle)


# ──────────────────────────────────────────────
#  구역 설정 저장/로드
# ──────────────────────────────────────────────
ZONE_CFG = Path("zone_config.json")


def save_zone(pts: list):
    ZONE_CFG.write_text(json.dumps({"zone": pts}), encoding="utf-8")


def load_zone() -> list:
    if ZONE_CFG.exists():
        try:
            return json.loads(ZONE_CFG.read_text())["zone"]
        except Exception:
            pass
    return []


# ──────────────────────────────────────────────
#  화재 / 연기 감지 (HSV 색상 기반)
# ──────────────────────────────────────────────
FIRE_TH        = 0.005  # 화재 픽셀 비율 임계값 (움직임 필터 적용 기준)
SMOKE_TH       = 0.50   # 연기 절대 임계값 (적응형 기준선과 함께 사용)
FIRE_CONFIRM_F = 10     # 연속 N프레임 이상 감지 시에만 알람 (오탐 방지)
CEIL_RATIO     = 0.30   # 화면 상단 N% 는 천장 조명으로 간주, 화재 분석 제외


def detect_fire_smoke(frame: np.ndarray) -> tuple[bool, bool, np.ndarray, float, float]:
    """
    HSV 색상 분석으로 화재/연기 감지.
    Returns: (fire_candidate, smoke_candidate, debug_mask, fire_ratio, smoke_ratio)

    ※ 연속 프레임 확인(FIRE_CONFIRM_F)은 VisitorManager에서 수행.
    ※ smoke_candidate는 절대 임계값만 반환 — 적응형 판단은 VisitorManager에서 수행.
    """
    h, w = frame.shape[:2]

    # ── 천장 조명 제외: 상단 CEIL_RATIO 영역 마스킹 ──
    roi_top = int(h * CEIL_RATIO)
    roi = frame[roi_top:, :]          # 하단 70% 만 분석
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ── 화재: 주황~노랑~빨강 (채도·명도 완화, 더 넓은 화염 색 포괄) ──
    # 낮 밝기 형광등 아래에서도 감지되도록 채도 기준 낮춤 (150 → 110)
    fire_lo1 = np.array([0,  110, 140])   # 빨강~주황
    fire_hi1 = np.array([25, 255, 255])
    fire_lo2 = np.array([155, 110, 140])  # 보라~빨강 (역방향)
    fire_hi2 = np.array([180, 255, 255])
    fire_mask_roi = (cv2.inRange(hsv, fire_lo1, fire_hi1) |
                     cv2.inRange(hsv, fire_lo2, fire_hi2))

    # 노이즈 제거: morphology open (작은 점 제거)
    fire_mask_roi = cv2.morphologyEx(
        fire_mask_roi, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # 불꽃 형태 검증: 픽셀 덩어리(contour)가 충분히 커야 함 (조명 반사 제거)
    contours, _ = cv2.findContours(
        fire_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_fire_mask = np.zeros_like(fire_mask_roi)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:          # 300px 미만 작은 점 제거 (조명 반사)
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bh / (bw + 1e-6)
        if aspect < 0.3:        # 너무 납작한 가로 픽셀 → 조명 반사
            continue
        cv2.drawContours(valid_fire_mask, [cnt], -1, 255, -1)

    fire_ratio = cv2.countNonZero(valid_fire_mask) / max(valid_fire_mask.size, 1)

    # 전체 프레임 크기 기준 debug_mask 생성 (상단 빈 영역 포함)
    fire_mask_full = np.zeros((h, w), dtype=np.uint8)
    fire_mask_full[roi_top:, :] = valid_fire_mask

    # ── 연기: 낮은 채도 + 중간 명도 ──
    hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    smoke_lo = np.array([0,   0,  60])
    smoke_hi = np.array([180, 45, 175])
    smoke_mask = cv2.inRange(hsv_full, smoke_lo, smoke_hi)
    smoke_mask = cv2.morphologyEx(
        smoke_mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    smoke_ratio = cv2.countNonZero(smoke_mask) / max(smoke_mask.size, 1)

    return fire_ratio > FIRE_TH, smoke_ratio > SMOKE_TH, fire_mask_full, fire_ratio, smoke_ratio


# ──────────────────────────────────────────────
#  VisitorManager
# ──────────────────────────────────────────────
MIN_TRACK_FRAMES = 6   # 이 프레임 수 이상 연속 감지된 ID만 카운트

class VisitorManager:
    EVENT_FRAMES = 100

    def __init__(self, model_path="yolo11n.pt", line_ratio=0.25,
                 flip=False, conf=0.40, zone_pts: list = None):
        logger.info(f"YOLO 모델 로드: {model_path}")
        self.model = YOLO(model_path)

        self.line_ratio = line_ratio   # 수직선 위치 (0~1 = 좌~우)
        self.flip       = flip
        self.conf       = conf

        # 침입 구역 (polygon, [[x,y], ...])
        self.zone_pts: list[list[int]] = zone_pts or []

        # 트래킹
        self.prev_x: dict[int, float] = {}      # tid → 이전 중심 X
        self.track_frames: dict[int, int] = {}   # tid → 연속 감지 프레임 수
        self.counted_ids: set[int] = set()       # 이미 카운트된 ID
        self.inside_ids: set[int]  = set()       # 경계선 안쪽 ID
        self.intrude_ids: set[int] = set()       # 침입 구역 현재 ID
        self.intrude_frames: dict[int, int] = {} # tid → 침입 구역 머문 프레임 수
        self.theft_ids: set[int] = set()         # 도난 알람 이미 발동된 ID

        # 낙상 감지
        self.fall_detector = FallDetector()
        self.fall_count    = 0

        # 얼굴 인식 (직원 제외)
        self.face_manager  = FaceManager()

        # 도난 감지 (Gemini VLM)
        self.theft_detector = TheftDetector(zone_pts=zone_pts)

        # 카카오 알림
        self.kakao = KakaoNotifier()

        # 통계
        self.total_visitors  = 0
        self.current_people  = 0
        self.fire_count      = 0
        self.intrusion_count = 0
        self.theft_count     = 0

        # 이벤트 표시
        self._alerts: deque = deque(maxlen=4)   # (text, color, timer)
        self._event_timer: dict[str, int] = {}

        # 화재/연기 쿨다운 + 연속 프레임 카운터
        self._fire_cd      = 0
        self._smoke_cd     = 0
        self.FIRE_CD       = 150   # 프레임
        self._fire_consec  = 0    # 연속 화재 감지 프레임 수
        self._smoke_consec = 0    # 연속 연기 감지 프레임 수

        # 연기 적응형 기준선 (처음 60프레임은 배경 학습)
        self._smoke_history: deque = deque(maxlen=120)  # 최근 120프레임 비율 저장
        self._smoke_baseline = 0.0
        self._prev_gray: np.ndarray | None = None      # 움직임 필터용 이전 프레임

        logger.info(f"초기화 완료 | 수직 경계선={line_ratio:.0%} "
                    f"| 침입구역={len(self.zone_pts)}점")

    def _side(self, cx: float, line_x: int) -> str:
        """경계선 기준 INSIDE/OUTSIDE"""
        if not self.flip:
            return "INSIDE" if cx >= line_x else "OUTSIDE"
        else:
            return "INSIDE" if cx <= line_x else "OUTSIDE"

    def _is_entry(self, px, cx, lx):
        if not self.flip:
            return px < lx <= cx
        return px > lx >= cx

    def _is_exit(self, px, cx, lx):
        if not self.flip:
            return px > lx >= cx
        return px < lx <= cx

    def _in_zone(self, cx: int, cy: int) -> bool:
        if len(self.zone_pts) < 3:
            return False
        pts = np.array(self.zone_pts, dtype=np.int32)
        return cv2.pointPolygonTest(pts, (cx, cy), False) >= 0

    def _add_alert(self, text: str, color: tuple):
        self._alerts.append([text, color, self.EVENT_FRAMES])
        logger.info(f"[ALERT] {text}")

    # ── 메인 처리 ────────────────────────────────
    def process(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        line_x = int(w * self.line_ratio)

        # ── 1) YOLO 트래킹 ──
        results = self.model.track(
            frame, persist=True, classes=[0],
            conf=self.conf, iou=0.45,
            tracker="bytetrack.yaml", verbose=False,
        )

        current_ids: set[int] = set()
        boxes_info: list[tuple] = []

        if (results[0].boxes is not None
                and results[0].boxes.id is not None):
            xywh  = results[0].boxes.xywh.cpu().numpy()
            ids   = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            # keypoints (yolov8n-pose 사용 시에만 존재)
            kps_all = None
            if results[0].keypoints is not None:
                kps_all = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)

            for idx, (box, tid, cf) in enumerate(zip(xywh, ids, confs)):
                cx_f, cy_f = float(box[0]), float(box[1])
                bw, bh = float(box[2]), float(box[3])
                cx_i, cy_i = int(cx_f), int(cy_f)
                bx1, by1 = int(cx_f - bw/2), int(cy_f - bh/2)

                current_ids.add(tid)
                boxes_info.append((bx1, by1, int(bw), int(bh), tid, cf))

                # 연속 감지 프레임 누적
                self.track_frames[tid] = self.track_frames.get(tid, 0) + 1

                # ── 경계선 통과 (MIN_TRACK_FRAMES 이후만 카운트) ──
                confirmed = self.track_frames[tid] >= MIN_TRACK_FRAMES

                if tid in self.prev_x:
                    px = self.prev_x[tid]
                    if confirmed and self._is_entry(px, cx_f, line_x):
                        if tid not in self.counted_ids:
                            self.counted_ids.add(tid)
                            self.inside_ids.add(tid)
                            self.total_visitors += 1
                            msg = f"어서오세요! (방문객 #{self.total_visitors})"
                            self._add_alert(msg, (0, 220, 60))
                            play_enter()
                            log_event("입장", f"ID={tid} 누적={self.total_visitors}")

                    elif confirmed and self._is_exit(px, cx_f, line_x):
                        if tid in self.inside_ids:
                            self.inside_ids.discard(tid)
                            self.counted_ids.discard(tid)  # 재입장 허용
                            self._add_alert("안녕히가세요!", (60, 160, 255))
                            play_exit()
                            log_event("퇴장", f"ID={tid}")
                else:
                    # 첫 감지 시 이미 INSIDE → 카운트
                    if (confirmed and
                            self._side(cx_f, line_x) == "INSIDE" and
                            tid not in self.counted_ids):
                        self.counted_ids.add(tid)
                        self.inside_ids.add(tid)
                        self.total_visitors += 1
                        msg = f"어서오세요! (방문객 #{self.total_visitors})"
                        self._add_alert(msg, (0, 220, 60))
                        play_enter()
                        log_event("입장", f"ID={tid}(첫감지) 누적={self.total_visitors}")

                self.prev_x[tid] = cx_f

                # ── 직원 여부 판단 (얼굴인식) ──
                face_bbox = (bx1, by1, bx1+int(bw), by1+int(bh))
                is_staff = self.face_manager.is_staff(frame, face_bbox)

                # ── 도난 감지 (Gemini VLM) ──
                theft_result = self.theft_detector.update(
                    tid, cx_i, cy_i, frame, is_staff)
                if theft_result:
                    self.theft_count += 1
                    msg = f"🚨 도난 의심! (ID:{tid})"
                    self._add_alert(msg, (0, 0, 200))
                    play_theft_alarm()
                    speak("도난이 의심됩니다 즉시 확인하세요")
                    log_event("도난의심", f"ID={tid} Gemini판단")
                    self.kakao.send("도난의심",
                                    f"행거 구역에서 도난 의심 행동 감지 (ID:{tid})",
                                    theft_result["frame"])

                # ── 낙상 감지 ──
                kps = kps_all[idx] if kps_all is not None and idx < len(kps_all) else None
                if self.fall_detector.update(tid, kps, bw, bh):
                    self.fall_count += 1
                    self._add_alert(f"🚨 낙상 감지! (ID:{tid})", (0, 0, 230))
                    speak("낙상이 감지되었습니다 즉시 확인하세요")
                    log_event("낙상감지", f"ID={tid}")
                    self.kakao.send("낙상감지", f"매장 내 낙상 감지 (ID:{tid})", frame)

                # ── 침입 구역 감지 ──
                in_z = self._in_zone(cx_i, cy_i)
                if in_z:
                    if tid not in self.intrude_ids:
                        self.intrude_ids.add(tid)
                        self.intrude_frames[tid] = 0
                        self.intrusion_count += 1
                        self._add_alert(f"⚠ 침입 감지! (ID:{tid})", (0, 80, 255))
                        speak("침입이 감지되었습니다")
                        log_event("침입감지", f"ID={tid}")
                    else:
                        self.intrude_frames[tid] = self.intrude_frames.get(tid, 0) + 1
                        # 침입 구역에 30프레임(≈3초) 이상 머물면 도난 의심 알람
                        if (self.intrude_frames[tid] >= 30
                                and tid not in self.theft_ids):
                            self.theft_ids.add(tid)
                            self.theft_count += 1
                            self._add_alert(f"🚨 도난 의심! (ID:{tid})", (0, 0, 255))
                            play_theft_alarm()
                            log_event("도난의심", f"ID={tid}")
                else:
                    self.intrude_ids.discard(tid)
                    self.intrude_frames.pop(tid, None)
                    self.theft_ids.discard(tid)

        self.current_people = len(current_ids)

        # 사라진 ID 정리
        lost = set(self.prev_x) - current_ids
        for tid in lost:
            del self.prev_x[tid]
            self.track_frames.pop(tid, None)
            self.inside_ids.discard(tid)
            self.intrude_ids.discard(tid)
            self.intrude_frames.pop(tid, None)
            self.theft_ids.discard(tid)
        self.fall_detector.cleanup(current_ids)
        self.theft_detector.cleanup(current_ids)

        # ── 2) 화재/연기 감지 (색상 + 움직임 필터) ──
        fire_color, _, _, _, smoke_r = detect_fire_smoke(frame)

        # 움직임 필터: 화재 색상 영역이 실제로 움직이는지 확인
        h_f = frame.shape[0]
        roi_top_f = int(h_f * CEIL_RATIO)
        gray_roi_f = cv2.cvtColor(frame[roi_top_f:], cv2.COLOR_BGR2GRAY)
        gray_roi_f = cv2.GaussianBlur(gray_roi_f, (5, 5), 0)
        if self._prev_gray is not None and fire_color:
            diff = cv2.absdiff(self._prev_gray, gray_roi_f)
            _, motion = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            motion_ratio = cv2.countNonZero(motion) / max(motion.size, 1)
            fire = fire_color and motion_ratio > 0.005  # 움직임 0.5% 이상이어야 불꽃
        else:
            fire = False  # 첫 프레임은 비교 불가
        self._prev_gray = gray_roi_f

        # 연기 적응형 판단: 배경 기준선 대비 급증 여부
        self._smoke_history.append(smoke_r)
        if len(self._smoke_history) < 60:
            # 처음 60프레임은 배경 학습 중 → 오탐 방지
            smoke = False
        else:
            # 기준선 = 하위 40% 값의 평균 (배경의 "정상" 연기 비율)
            sorted_h = sorted(self._smoke_history)
            n40 = max(1, len(sorted_h) * 2 // 5)
            self._smoke_baseline = sum(sorted_h[:n40]) / n40
            # 기준선보다 15%p 이상 급증 + 최소 20% 이상일 때만 연기로 판단
            # (벽이 평소에 40% 회색이면 → 55%+ 되어야 트리거)
            smoke = (smoke_r > self._smoke_baseline + 0.15 and smoke_r > 0.20)

        # ── 연속 프레임 카운터 업데이트 ──
        self._fire_consec  = self._fire_consec + 1  if fire  else 0
        self._smoke_consec = self._smoke_consec + 1 if smoke else 0

        # ── 화재 알람: 연속 FIRE_CONFIRM_F 프레임 이상 + 쿨다운 ──
        if self._fire_consec >= FIRE_CONFIRM_F and self._fire_cd == 0:
            self._fire_cd = self.FIRE_CD
            self._fire_consec = 0
            self.fire_count += 1
            self._add_alert("🔥 화재 감지!", (0, 40, 220))
            play_fire_alarm()
            speak("화재가 감지되었습니다 즉시 대피하세요")
            log_event("화재감지")
            self.kakao.send("화재감지", "매장 내 화재 감지! 즉시 대피하세요", frame)

        if self._smoke_consec >= FIRE_CONFIRM_F and self._fire_cd == 0:
            self._fire_cd = self.FIRE_CD
            self._smoke_consec = 0
            self.fire_count += 1
            self._add_alert("💨 연기 감지!", (60, 60, 200))
            play_fire_alarm()
            speak("연기가 감지되었습니다")
            log_event("연기감지")
            self.kakao.send("연기감지", "매장 내 연기 감지!", frame)
        if self._fire_cd > 0:
            self._fire_cd -= 1

        # ── 3) 화면 그리기 ──
        frame = self._draw(frame, line_x, boxes_info, w, h)

        # 알람 타이머
        for a in self._alerts:
            a[2] = max(0, a[2] - 1)

        return frame

    # ── 그리기 ──────────────────────────────────
    def _draw(self, frame, line_x, boxes_info, w, h):

        # ① 침입 구역 반투명 채우기
        if len(self.zone_pts) >= 3:
            pts = np.array(self.zone_pts, np.int32)
            ov = frame.copy()
            cv2.fillPoly(ov, [pts], (0, 60, 220))
            cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
            cv2.polylines(frame, [pts], True, (0, 100, 255), 2)
            cx_z = int(pts[:, 0].mean())
            cy_z = int(pts[:, 1].mean())
            frame = kr(frame, "침입 금지 구역", (cx_z - 60, cy_z - 12),
                       F_SM if USE_PIL else None, (0, 140, 255))

        # ② 수직 경계선 (띠 + 선)
        ov2 = frame.copy()
        cv2.rectangle(ov2, (line_x - 18, 0), (line_x + 18, h), (0, 200, 80), -1)
        cv2.addWeighted(ov2, 0.18, frame, 0.82, 0, frame)
        cv2.line(frame, (line_x, 0), (line_x, h), (0, 230, 80), 3)
        cv2.line(frame, (line_x - 1, 0), (line_x - 1, h), (255, 255, 255), 1)
        cv2.line(frame, (line_x + 1, 0), (line_x + 1, h), (255, 255, 255), 1)

        # 방향 표시
        ent_x = line_x + 30 if not self.flip else line_x - 90
        ext_x = line_x - 90 if not self.flip else line_x + 30
        frame = kr(frame, "▶ 입장", (ent_x, h // 2 - 28),
                   F_SM if USE_PIL else None, (0, 230, 80))
        frame = kr(frame, "◀ 퇴장", (ext_x, h // 2 + 10),
                   F_SM if USE_PIL else None, (60, 160, 255))
        frame = kr(frame, "출입 경계선",
                   (line_x - 42, 10), F_SM if USE_PIL else None, (255, 255, 255))

        # ③ 바운딩 박스
        for (x1, y1, bw, bh, tid, cf) in boxes_info:
            fallen = self.fall_detector.is_fallen(tid)
            if fallen:
                color = (30, 0, 255)   # 진빨강 = 낙상
            elif tid in self.intrude_ids:
                color = (0, 60, 255)   # 빨강 = 침입
            elif tid in self.inside_ids:
                color = (0, 210, 60)   # 초록 = 안쪽
            else:
                color = (200, 200, 50) # 노랑 = 바깥
            thickness = 3 if fallen else 2
            cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), color, thickness)
            fall_str   = " [낙상!]" if fallen else ""
            frames_str = f"({self.track_frames.get(tid,0)}f)"
            cv2.putText(frame, f"ID:{tid} {cf:.2f} {frames_str}{fall_str}",
                        (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        # ④ 정보 패널 (상단)
        ov3 = frame.copy()
        cv2.rectangle(ov3, (0, 0), (w, 115), (0, 0, 0), -1)
        cv2.addWeighted(ov3, 0.55, frame, 0.45, 0, frame)

        panel_lines = [
            (f"현재 인원 : {self.current_people} 명",         (255, 255, 255), 10),
            (f"오늘 누적 방문객 : {self.total_visitors} 명",  (0, 220, 255),   44),
            (f"침입 {self.intrusion_count}건 | 화재/연기 {self.fire_count}건 | 낙상 {self.fall_count}건",
             (200, 140, 60), 76),
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),    (140, 140, 140), 102),
        ]
        for text, color, y in panel_lines:
            frame = kr(frame, text, (14, y), F_MD if USE_PIL else None, color)

        # ⑤ 알람 메시지 (하단 스택)
        active = [(t, c, tm) for t, c, tm in self._alerts if tm > 0]
        for i, (text, color, timer) in enumerate(reversed(active[-3:])):
            yy = h - 38 - i * 44
            alpha = min(timer / 30, 1.0)
            a_color = tuple(int(c * alpha) for c in color)
            frame = kr(frame, text, (w // 2 - 200, yy),
                       F_LG if USE_PIL else None, a_color)

        # ⑥ 조작 안내
        cv2.putText(frame, "Q:quit  U/D:line  Z:zone  R:report  F:flip",
                    (w - 390, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (120, 120, 120), 1, cv2.LINE_AA)

        return frame


# ──────────────────────────────────────────────
#  마우스 콜백 (구역 설정)
# ──────────────────────────────────────────────
_zone_drawing  = False
_zone_tmp: list[list[int]] = []


def mouse_cb(event, x, y, flags, param):
    global _zone_drawing, _zone_tmp
    if not _zone_drawing:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        _zone_tmp.append([x, y])


# ──────────────────────────────────────────────
#  argparse
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0")
    p.add_argument("--model",  default="yolo11n.pt")
    p.add_argument("--line",   type=float, default=0.20,
                   help="수직 경계선 위치 (0~1, 기본 0.20 = 좌측 20%%)")
    p.add_argument("--flip",   action="store_true")
    p.add_argument("--conf",   type=float, default=0.40)
    p.add_argument("--width",  type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


# ──────────────────────────────────────────────
#  main
# ──────────────────────────────────────────────
def main():
    global _zone_drawing, _zone_tmp

    args = parse_args()

    # TTS 스레드
    tts_t = threading.Thread(target=_tts_worker, daemon=True)
    tts_t.start()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"카메라 열기 실패: {source}")
        sys.exit(1)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

    zone_pts = load_zone()
    vm = VisitorManager(model_path=args.model, line_ratio=args.line,
                        flip=args.flip, conf=args.conf, zone_pts=zone_pts)

    WIN = "지능형 방문객 관리 시스템 v2.0"

    # ── GUI 초기화 (헤드리스 환경에서는 건너뜀) ──
    if not HEADLESS:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, mouse_cb)

    print("\n" + "=" * 60)
    print("  지능형 매장 방문객 관리 시스템  v2.0")
    if HEADLESS:
        print("  [헤드리스 모드] 디스플레이 없음 — 콘솔 출력만")
        print("  종료: Ctrl+C")
    else:
        print("  조작키: Q=종료  U/D=선이동  Z=구역설정  R=보고서  F=방향전환")
    print("=" * 60)
    print(f"  소스   : {source}")
    print(f"  경계선 : 수직선 좌측 {args.line:.0%}")
    print("=" * 60 + "\n")

    fps_t = time.time()
    fcnt  = 0
    fps   = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = vm.process(frame)

            # FPS 계산 및 콘솔 출력
            fcnt += 1
            elapsed = time.time() - fps_t
            if elapsed >= 1.0:
                fps = fcnt / elapsed
                fcnt = 0; fps_t = time.time()
                print(f"\r현재:{vm.current_people}명 | 누적:{vm.total_visitors}명 "
                      f"| 침입:{vm.intrusion_count} | 화재:{vm.fire_count} | FPS:{fps:.1f}",
                      end="", flush=True)

            if HEADLESS:
                # 헤드리스: imshow 없이 짧은 sleep 후 다음 프레임
                time.sleep(0.005)
                continue

            # ── GUI 모드 (모니터 연결 PC) ──
            # 구역 설정 모드 오버레이
            if _zone_drawing:
                h, w = frame.shape[:2]
                cv2.putText(frame,
                            f"[구역설정] 클릭({len(_zone_tmp)}개) | Enter=확정 | Esc=취소",
                            (10, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 2)
                for pt in _zone_tmp:
                    cv2.circle(frame, tuple(pt), 5, (0, 200, 255), -1)
                if len(_zone_tmp) >= 2:
                    cv2.polylines(frame, [np.array(_zone_tmp, np.int32)],
                                  False, (0, 200, 255), 2)

            cv2.putText(frame, f"FPS:{fps:.1f}" if elapsed < 2 else "",
                        (frame.shape[1] - 90, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('u'):
                vm.line_ratio = max(0.05, vm.line_ratio - 0.02)
            elif key == ord('d'):
                vm.line_ratio = min(0.95, vm.line_ratio + 0.02)
            elif key == ord('f'):
                vm.flip = not vm.flip
            elif key == ord('r'):
                generate_report(vm.total_visitors, vm.fire_count, vm.intrusion_count)
                print("\n보고서 생성 완료!")
            elif key == ord('z'):
                if not _zone_drawing:
                    _zone_drawing = True; _zone_tmp = []
                    print("\n[구역설정] 마우스 클릭으로 꼭짓점 추가 → Enter 확정")
                else:
                    _zone_drawing = False; _zone_tmp = []
                    print("\n[구역설정] 취소")
            elif key == 13 and _zone_drawing:   # Enter
                if len(_zone_tmp) >= 3:
                    vm.zone_pts = _zone_tmp.copy()
                    save_zone(_zone_tmp)
                    print(f"\n[구역설정] 확정: {len(_zone_tmp)}개 꼭짓점 저장")
                _zone_drawing = False; _zone_tmp = []
            elif key == 27 and _zone_drawing:   # Esc
                _zone_drawing = False; _zone_tmp = []

    except KeyboardInterrupt:
        print("\n\n[Ctrl+C] 종료 요청")

    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()

    # 종료 시 보고서 자동 생성
    print("\n\n보고서 생성 중...")
    rpt = generate_report(vm.total_visitors, vm.fire_count, vm.intrusion_count)

    _tts_q.put(None)
    tts_t.join(timeout=2)

    print(f"\n{'=' * 50}")
    print(f"  총 방문객 : {vm.total_visitors} 명")
    print(f"  침입 감지 : {vm.intrusion_count} 건")
    print(f"  화재/연기 : {vm.fire_count} 건")
    print(f"  보고서    : {rpt}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
