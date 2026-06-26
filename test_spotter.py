import sys
import sherpa_onnx
import glob
import os

model_dir = "models/sherpa-onnx"

def _pick(pattern):
    files = glob.glob(pattern)
    return [f for f in files if "int8" not in os.path.basename(f)] or files

_enc = _pick(os.path.join(model_dir, "encoder-*.onnx"))
_dec = _pick(os.path.join(model_dir, "decoder-*.onnx"))
_joi = _pick(os.path.join(model_dir, "joiner-*.onnx"))
tokens = os.path.join(model_dir, "tokens.txt")
keywords = os.path.join(model_dir, "keywords.txt")

spotter = sherpa_onnx.KeywordSpotter(
    tokens=tokens,
    encoder=_enc[0],
    decoder=_dec[0],
    joiner=_joi[0],
    keywords_file=keywords,
    keywords_threshold=0.2,
    num_threads=1,
)

stream = spotter.create_stream()
res = spotter.get_result(stream)
print("Type of result:", type(res))
print("Result truthiness:", bool(res))
print("Result string repr:", repr(res))
if hasattr(res, 'keyword'):
    print("Keyword:", repr(res.keyword))
