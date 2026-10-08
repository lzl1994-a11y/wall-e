"""Pure wrapper regressions plus opt-in X3 hardware/codec integration."""
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch
import pytest
from services.remote.x3_video import X3VideoCodec, HardwareVideoError, require_codec_library


def sdk():
    lib = Mock()
    for name in ["walle_init", "walle_close", "walle_decode", "walle_encoder_close", "walle_encoder_open"]:
        getattr(lib, name).return_value = 0
    lib.walle_encode.return_value = 5
    return lib


def test_native_init_failure_releases_owner_and_partial_resources():
    lib = sdk(); lib.walle_init.return_value = -17
    with patch("services.remote.x3_video.ctypes.CDLL", return_value=lib):
        with pytest.raises(HardwareVideoError, match="initialize failed: -17"):
            X3VideoCodec(Path("test.so"))
        lib.walle_close.assert_called_once()
        lib.walle_init.return_value = 0
        codec = X3VideoCodec(Path("test.so"))
        codec.close(); codec.close()
    assert lib.walle_close.call_count == 2


def test_second_session_cannot_reset_active_sdk():
    lib = sdk()
    with patch("services.remote.x3_video.ctypes.CDLL", return_value=lib):
        first = X3VideoCodec(Path("test.so"))
        try:
            with pytest.raises(HardwareVideoError, match="already in use"):
                X3VideoCodec(Path("test.so"))
            lib.walle_init.assert_called_once()
        finally:
            first.close()


def test_close_failure_still_releases_owner():
    lib = sdk()
    with patch("services.remote.x3_video.ctypes.CDLL", return_value=lib):
        codec = X3VideoCodec(Path("test.so"));lib.walle_close.return_value = -5
        with pytest.raises(HardwareVideoError, match="close failed: -5"):codec.close()
        lib.walle_close.return_value = 0
        replacement = X3VideoCodec(Path("test.so")); replacement.close()


def test_missing_native_build_is_explicit():
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(HardwareVideoError, match="build_webrtc_codec.sh"):
            require_codec_library(Path(directory))


def test_profile_switch_reopens_encoder_and_rejects_bad_input():
    Image = pytest.importorskip("PIL.Image")
    from io import BytesIO
    lib = sdk()
    out = BytesIO();Image.new("RGB", (480,360), "red").save(out,format="JPEG")
    with patch("services.remote.x3_video.ctypes.CDLL", return_value=lib):
        codec = X3VideoCodec(Path("test.so"))
        try:
            with pytest.raises(ValueError):codec.encode(b"",480,360,10,0)
            codec.encode(out.getvalue(),480,360,10,9000)
            codec.encode(out.getvalue(),480,360,10,18000)
            assert lib.walle_encoder_open.call_count == 1
            lib.walle_encoder_open.assert_called_with(480,360,10,500)
            lib.walle_encode.return_value = -9
            with pytest.raises(HardwareVideoError,match="H264 encode failed: -9"):
                codec.encode(out.getvalue(),480,360,5,27000)
            lib.walle_encoder_open.assert_called_with(480,360,5,150)
        finally:codec.close()


def test_real_x3_jpeg_encode_switch_orientation_and_release():
    import os
    if os.environ.get("WALLE_TEST_X3_VIDEO") != "1":
        pytest.skip("opt-in X3 SDK hardware test")
    import av, numpy as np
    from PIL import Image
    from io import BytesIO
    image = Image.new("RGB", (640,480), "red")
    image.paste("lime",(320,0,640,240));image.paste("blue",(0,240,320,480));image.paste("white",(320,240,640,480))
    stream=BytesIO();image.save(stream,format="JPEG",quality=95)
    jpeg=stream.getvalue()
    library=require_codec_library(Path(__file__).resolve().parents[1])
    for _ in range(2):
        codec=X3VideoCodec(library)
        try:
            for w,h,fps in [(480,360,10),(320,240,5),(480,360,10)]:
                bitstream=b''.join(codec.encode(jpeg,w,h,fps,i*9000) for i in range(12))
                # Checking only SPS profile misses SDK's default CABAC PPS,
                # which software decoders accept but Baseline browsers cannot.
                from aiortc.codecs.h264 import H264Encoder
                nals=list(H264Encoder._split_bitstream(bitstream))
                sps=next(n for n in nals if n[0]&31==7)
                assert sps[1:4].hex()=='42001f'
                for pps in (n for n in nals if n[0]&31==8):
                    rbsp=pps[1:].replace(b'\x00\x00\x03',b'\x00\x00')
                    bits=''.join(f'{b:08b}' for b in rbsp);offset=0
                    for _ in range(2): # pic_parameter_set_id, seq_parameter_set_id
                        zeros=0
                        while bits[offset]=='0':zeros+=1;offset+=1
                        offset+=1+zeros
                    assert bits[offset]=='0', 'Baseline must use CAVLC, not CABAC'
                decoder=av.CodecContext.create('h264','r');frames=[]
                for packet in decoder.parse(bitstream):frames.extend(decoder.decode(packet))
                for packet in decoder.parse(b''):frames.extend(decoder.decode(packet))
                frames.extend(decoder.decode(None));assert len(frames)==12
                for frame in frames:
                    assert (frame.width,frame.height)==(w,h)
                    p=frame.to_ndarray(format='rgb24')
                    assert int(np.argmax(p[h//4,w//4]))==2 # top-left blue after vertical flip
                    assert p[h//4,w*3//4].mean()>180 # top-right white
                    assert int(np.argmax(p[h*3//4,w//4]))==0 # bottom-left red
                    assert int(np.argmax(p[h*3//4,w*3//4]))==1 # bottom-right green
        finally:codec.close()
