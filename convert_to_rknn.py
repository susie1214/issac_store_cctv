"""
ONNX → RKNN 변환 스크립트 (OrangePi 5 Ultra / RK3588)
=======================================================
변환 전 vs 후 비교:
  ONNX + onnxruntime  : CPU 추론 → ~8-15 FPS (RK3588 4코어)
  RKNN + rknn-toolkit2: NPU 추론 → ~40-80 FPS (RK3588 NPU 6 TOPS)

변환 순서:
  1. 변환은 PC(x86 Ubuntu 20.04/22.04)에서 진행 — OrangePi에서는 불가
  2. 변환된 .rknn 파일을 OrangePi로 복사
  3. OrangePi에서 rknn-toolkit-lite2 + rknnlite로 추론

준비 (PC에서):
  pip install rknn-toolkit2          # Rockchip 공식 PyPI
  # 또는 .whl 직접 설치:
  # pip install rknn_toolkit2-2.x.x-cp310-cp310-linux_x86_64.whl

준비 (OrangePi에서):
  pip install rknn-toolkit-lite2     # 경량 추론 전용

실행:
  python convert_to_rknn.py                  # yolov8n-pose.onnx 변환 (이것만 있으면 됨)
  python convert_to_rknn.py --no-quantize    # FP16 모드 (calibration 이미지 없어도 OK)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── 변환할 모델 목록 ──────────────────────────────
MODELS = {
    # yolov8n-pose: 사람 감지 + 17개 관절 keypoint 동시 출력
    # → 피플카운팅 + 낙상감지 모두 처리 (이 모델 하나로 충분)
    "yolov8n-pose": {
        "onnx":         "yolov8n-pose.onnx",
        "rknn":         "models/yolov8n_pose.rknn",
        "input_size":   [1, 3, 640, 640],
        "mean_values":  [[0, 0, 0]],
        "std_values":   [[255, 255, 255]],
    },
}

# RK3588 NPU 플랫폼 코드
PLATFORM = "rk3588"


def convert(model_key: str, quantize: bool = True) -> Path:
    """
    ONNX → RKNN 변환 실행.

    quantize=True  : INT8 양자화 (속도 ↑, 정확도 약간 ↓) — 권장
    quantize=False : FP16 모드 (정확도 유지, 속도 약간 ↓)
    """
    cfg = MODELS[model_key]
    onnx_path = Path(cfg["onnx"])
    rknn_path = Path(cfg["rknn"])

    if not onnx_path.exists():
        print(f"[오류] ONNX 파일 없음: {onnx_path}")
        return None

    rknn_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from rknn.api import RKNN
    except ImportError:
        print("[오류] rknn-toolkit2 미설치")
        print("  → pip install rknn-toolkit2")
        print("  ※ 변환은 x86 PC(Ubuntu)에서만 가능합니다")
        sys.exit(1)

    rknn = RKNN(verbose=False)

    # ── 1) 설정 ──
    print(f"\n[{model_key}] 변환 설정...")
    rknn.config(
        mean_values=cfg["mean_values"],
        std_values=cfg["std_values"],
        target_platform=PLATFORM,
        quantized_algorithm="normal",   # normal / mmse / kl_divergence
        quantized_method="channel",     # channel(권장) / layer
        optimization_level=3,           # 0~3, 높을수록 최적화 강도 증가
    )

    # ── 2) ONNX 로드 ──
    print(f"[{model_key}] ONNX 로드: {onnx_path}")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        print(f"[오류] ONNX 로드 실패 (코드: {ret})")
        return None

    # ── 3) 빌드 (양자화 포함) ──
    quant_str = "INT8 양자화" if quantize else "FP16"
    print(f"[{model_key}] 빌드 ({quant_str})...")
    ret = rknn.build(do_quantization=quantize, dataset="dataset.txt")
    if ret != 0:
        print(f"[오류] 빌드 실패 (코드: {ret})")
        print("  ※ dataset.txt 가 없으면 quantize=False 로 실행하세요")
        return None

    # ── 4) 저장 ──
    print(f"[{model_key}] 저장: {rknn_path}")
    ret = rknn.export_rknn(str(rknn_path))
    if ret != 0:
        print(f"[오류] 저장 실패 (코드: {ret})")
        return None

    rknn.release()
    print(f"[{model_key}] 완료 ✅  → {rknn_path}")
    return rknn_path


def make_dataset_txt(n_images: int = 20):
    """
    양자화용 calibration 이미지 목록 생성.
    static/ 또는 현재 폴더에서 jpg/png를 n개 수집.
    """
    import glob
    imgs = (glob.glob("static/**/*.jpg", recursive=True) +
            glob.glob("static/**/*.png", recursive=True) +
            glob.glob("*.jpg") + glob.glob("*.png"))
    imgs = imgs[:n_images]

    if not imgs:
        print("[경고] calibration 이미지 없음 → FP16 모드로 전환 권장")
        return False

    Path("dataset.txt").write_text("\n".join(imgs))
    print(f"dataset.txt 생성 ({len(imgs)}개 이미지)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="ONNX → RKNN 변환 (OrangePi RK3588용)",
    )
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        default="yolov8n-pose", help="변환할 모델 (기본: yolov8n-pose)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="INT8 양자화 건너뜀 (FP16 모드)")
    args = parser.parse_args()

    quantize = not args.no_quantize

    # 양자화 시 calibration 데이터 준비
    if quantize:
        ok = make_dataset_txt()
        if not ok:
            print("  → --no-quantize 옵션으로 재실행하거나")
            print("     dataset.txt 에 이미지 경로를 직접 입력하세요")
            quantize = False
            print("  자동으로 FP16 모드로 전환합니다.\n")

    targets = [args.model]
    results = {}
    for key in targets:
        out = convert(key, quantize=quantize)
        results[key] = out

    print("\n" + "=" * 50)
    print("  변환 결과")
    print("=" * 50)
    for key, path in results.items():
        status = f"✅ {path}" if path else "❌ 실패"
        print(f"  {key:20s} : {status}")
    print("=" * 50)
    print()
    print("다음 단계:")
    print("  1. 생성된 models/*.rknn 파일을 OrangePi 로 복사")
    print("     scp models/*.rknn orangepi@172.30.1.34:~/visitor-management/models/")
    print()
    print("  2. OrangePi에서 rknn-toolkit-lite2 설치")
    print("     pip install rknn-toolkit-lite2")
    print()
    print("  3. visitor_manager.py 에서 RKNN 추론 활성화")
    print("     --model models/yolo11n.rknn  (확장자로 자동 감지)")


if __name__ == "__main__":
    main()
