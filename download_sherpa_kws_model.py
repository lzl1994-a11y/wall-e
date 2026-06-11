#!/usr/bin/env python3
"""下载 Sherpa-ONNX 中文唤醒词模型 (wenetspeech 3.3M)

模型: sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01
体积: ~13MB (tar.bz2)
目标: models/sherpa-onnx/

需要的文件:
  - tokens.txt
  - encoder-epoch-99-avg-1.onnx
  - decoder-epoch-99-avg-1.onnx
  - joiner-epoch-99-avg-1.onnx
  - keywords.txt (已存在，不会被覆盖)
"""

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

MODEL_DIR = Path(__file__).resolve().parent / "models" / "sherpa-onnx"


def download_with_progress(url, dest):
    """带进度条的下载。"""
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
    """解压 tar.bz2，只提取需要的 4 个模型文件。"""
    needed = {
        "tokens.txt",
        "encoder-epoch-99-avg-1.onnx",
        "decoder-epoch-99-avg-1.onnx",
        "joiner-epoch-99-avg-1.onnx",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"解压到: {MODEL_DIR}")
    with tarfile.open(archive_path, "r:bz2") as tar:
        for member in tar.getmembers():
            basename = os.path.basename(member.name)
            if basename in needed:
                # 去掉前缀目录，直接解压到 MODEL_DIR
                member.name = basename
                tar.extract(member, MODEL_DIR)
                print(f"  ✓ {basename}")


def main():
    if MODEL_DIR.exists():
        existing = list(MODEL_DIR.glob("*.onnx")) + list(MODEL_DIR.glob("tokens.txt"))
        if len(existing) >= 4:
            print(f"模型文件已存在: {MODEL_DIR}")
            print("跳过下载（如需重新下载请手动删除该目录）")
            return

    archive = MODEL_DIR.parent / "sherpa-onnx-kws.tar.bz2"

    try:
        download_with_progress(MODEL_URL, str(archive))
    except Exception as e:
        print(f"\n下载失败: {e}")
        print("\n手动下载步骤:")
        print(f"  1. 打开浏览器访问: {MODEL_URL}")
        print(f"  2. 下载后解压，将以下文件放入 {MODEL_DIR}:")
        print("     - tokens.txt")
        print("     - encoder-epoch-99-avg-1.onnx")
        print("     - decoder-epoch-99-avg-1.onnx")
        print("     - joiner-epoch-99-avg-1.onnx")
        return

    extract_model(archive)

    # 清理
    os.remove(archive)

    # 确保 keywords.txt 存在
    keywords_file = MODEL_DIR / "keywords.txt"
    if not keywords_file.exists():
        keywords_file.write_text("wa li wa li @瓦力瓦力", encoding="utf-8")
        print(f"  ✓ keywords.txt (自动生成)")

    print(f"\n完成! 模型已就绪: {MODEL_DIR}")


if __name__ == "__main__":
    main()
