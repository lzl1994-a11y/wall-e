#!/usr/bin/env python3
"""下载 Sherpa-ONNX 中文唤醒词模型 (wenetspeech 3.3M)

模型: sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01
体积: ~13MB (tar.bz2)
目标: models/sherpa-onnx/

需要的文件（自动匹配，不硬编码版本号）:
  - tokens.txt
  - encoder-*.onnx
  - decoder-*.onnx
  - joiner-*.onnx
  - keywords.txt (已存在不会被覆盖)
"""

import glob as glob_mod
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "kws-models/"
    "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2"
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "sherpa-onnx"


def download_with_progress(url, dest):
    print(f"下载: {url}")
    print(f"保存到: {dest}")

    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size * 100 / total_size, 100)
            downloaded = count * block_size / (1024 * 1024)
            total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {pct:.0f}%  {downloaded:.1f}/{total:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def extract_model(archive_path):
    """解压 tar.bz2，按文件名模式匹配提取模型文件。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"解压到: {MODEL_DIR}")
    extracted = []

    with tarfile.open(archive_path, "r:bz2") as tar:
        for member in tar.getmembers():
            basename = os.path.basename(member.name)
            # 按前缀匹配，不硬编码版本号
            if (basename.startswith("encoder-") and basename.endswith(".onnx")) or \
               (basename.startswith("decoder-") and basename.endswith(".onnx")) or \
               (basename.startswith("joiner-") and basename.endswith(".onnx")) or \
               basename == "tokens.txt":
                member.name = basename
                tar.extract(member, MODEL_DIR)
                extracted.append(basename)
                print(f"  ✓ {basename}")

    if not extracted:
        print("  ⚠ 未找到任何模型文件，压缩包内容如下:")
        with tarfile.open(archive_path, "r:bz2") as tar:
            for member in tar.getmembers():
                print(f"    {member.name}")
        return False

    return True


def main():
    # 如果模型已存在，跳过
    existing_onnx = list(MODEL_DIR.glob("encoder-*.onnx")) + \
                    list(MODEL_DIR.glob("decoder-*.onnx")) + \
                    list(MODEL_DIR.glob("joiner-*.onnx"))
    if existing_onnx and (MODEL_DIR / "tokens.txt").exists():
        print(f"模型文件已存在: {MODEL_DIR}")
        for f in sorted(MODEL_DIR.glob("*")):
            print(f"  {f.name}")
        return

    archive = MODEL_DIR.parent / "sherpa-onnx-kws.tar.bz2"

    try:
        download_with_progress(MODEL_URL, str(archive))
    except Exception as e:
        print(f"\n下载失败: {e}")
        print("\n手动下载步骤:")
        print(f"  浏览器打开: {MODEL_URL}")
        print(f"  下载后解压，将以下文件放入 {MODEL_DIR}:")
        print("    - tokens.txt")
        print("    - encoder-*.onnx")
        print("    - decoder-*.onnx")
        print("    - joiner-*.onnx")
        return

    ok = extract_model(archive)

    # 清理压缩包
    os.remove(archive)

    # keywords.txt
    keywords_file = MODEL_DIR / "keywords.txt"
    if not keywords_file.exists():
        keywords_file.write_text("wa li wa li @瓦力瓦力", encoding="utf-8")
        print(f"  ✓ keywords.txt (自动生成)")

    if ok:
        print(f"\n完成! 模型已就绪: {MODEL_DIR}")
        print("\n文件清单:")
        for f in sorted(MODEL_DIR.glob("*")):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
