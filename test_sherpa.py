import os
import sys

# Append the directory to sys.path so we can import sherpa_onnx if installed globally
try:
    import sherpa_onnx
    print("Sherpa-ONNX imported successfully.")
    
    # Check what type get_result might return on a dummy stream if possible
    # We can't easily instantiate KeywordSpotter without valid models.
    # But we can inspect the module.
    print("KeywordSpotter methods:", dir(sherpa_onnx.KeywordSpotter))
    print("KeywordResult methods:", dir(sherpa_onnx.KeywordResult) if hasattr(sherpa_onnx, 'KeywordResult') else "No KeywordResult found")
except ImportError:
    print("sherpa_onnx not installed in this environment.")
except Exception as e:
    print("Error:", e)
