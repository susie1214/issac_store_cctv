"""
모델 다운로드 스크립트
=====================
실행: python download_models.py

다운로드 파일:
  [나이/성별 ~70MB]
  - opencv_face_detector.pbtxt / .pb  (얼굴 감지)
  - age_deploy.prototxt / age_net.caffemodel
  - gender_deploy.prototxt / gender_net.caffemodel

  [소리 감지 ~2MB]
  - yamnet.tflite          (Google YAMNet — 비명·유리깨짐 등 521종)
  - yamnet_class_map.csv   (클래스 목록)
"""

import urllib.request
import sys
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FILES = [
    # 얼굴 감지 (OpenCV DNN)
    ("opencv_face_detector.pbtxt",
     "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"),
    ("opencv_face_detector_uint8.pb",
     "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb"),
    # 나이 분류 (Levi & Hassner Caffe)
    ("age_deploy.prototxt",
     "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net_definitions/deploy.prototxt"),
    ("age_net.caffemodel",
     "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel"),
    # 성별 분류
    ("gender_deploy.prototxt",
     "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/deploy.prototxt"),
    ("gender_net.caffemodel",
     "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel"),
    # YAMNet 소리 감지 (Google AudioSet, ~2MB)
    ("yamnet.tflite",
     "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/android/lite-model_yamnet_classification_tflite_1.tflite"),
    ("yamnet_class_map.csv",
     "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"),
]


def _progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100)
    bar = "█" * int(pct // 4) + "░" * (25 - int(pct // 4))
    mb = downloaded / 1024 / 1024
    total_mb = total_size / 1024 / 1024
    print(f"\r  [{bar}] {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)


def main():
    print("=" * 55)
    print("  모델 다운로드 (나이/성별 + YAMNet 소리감지)")
    print("=" * 55)
    for fname, url in FILES:
        dest = MODEL_DIR / fname
        if dest.exists():
            print(f"  ✅ {fname} (이미 있음, 건너뜀)")
            continue
        print(f"\n  ⬇  {fname}")
        try:
            urllib.request.urlretrieve(url, dest, _progress)
            print(f"\n  완료: {dest}")
        except Exception as e:
            print(f"\n  ❌ 다운로드 실패: {e}")
            print(f"     수동 다운로드: {url}")
            dest.unlink(missing_ok=True)

    # 검증
    print("\n" + "=" * 55)
    print("  검증")
    all_ok = True
    for fname, _ in FILES:
        ok = (MODEL_DIR / fname).exists()
        print(f"  {'✅' if ok else '❌'} {fname}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  모든 모델 준비 완료! python web_app.py 실행하세요.")
    else:
        print("\n  일부 파일 누락. 수동 다운로드 후 models/ 폴더에 저장하세요.")
        print("  소리감지 패키지: pip install sounddevice tflite-runtime")
    print("=" * 55)


if __name__ == "__main__":
    main()
