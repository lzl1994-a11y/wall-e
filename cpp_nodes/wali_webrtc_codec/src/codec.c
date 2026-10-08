/* X3 public media SDK bridge. Calls are serialized by one Python worker.
 * Decode/encode channels 2/3 are reserved for this gateway; AI uses channel 1.
 * No frame dumps, process-global hooks, or SDK source modifications.
 */
#include <stdint.h>
#include <sys/types.h>
#include <string.h>
#include "hb_vdec.h"
#include "hb_venc.h"
#include "hb_vp_api.h"

enum { DECODER = 2, ENCODER = 3, TIMEOUT_MS = 400, MAX_JPEG = 2097152 };
static int vp, enc_module, dec_module, enc_created, enc_started, dec_created, dec_started;
static int enc_width, enc_height;
static uint64_t enc_physical, jpeg_physical;
static void *enc_pixels, *jpeg_pixels;

/* Preserve the first cleanup error while attempting all owned releases. */
static void remember(int *result, int error) { if (!*result && error) *result = error; }
int walle_encoder_close(void) {
    int result = 0;
    if (enc_started) { remember(&result, HB_VENC_StopRecvFrame(ENCODER)); enc_started = 0; }
    if (enc_created) { remember(&result, HB_VENC_DestroyChn(ENCODER)); enc_created = 0; }
    if (enc_pixels) { remember(&result, HB_SYS_Free(enc_physical, enc_pixels)); enc_pixels = NULL; }
    return result;
}
int walle_close(void) {
    int result = walle_encoder_close();
    if (dec_started) { remember(&result, HB_VDEC_StopRecvStream(DECODER)); dec_started = 0; }
    if (dec_created) { remember(&result, HB_VDEC_DestroyChn(DECODER)); dec_created = 0; }
    if (jpeg_pixels) { remember(&result, HB_SYS_Free(jpeg_physical, jpeg_pixels)); jpeg_pixels = NULL; }
    if (dec_module) { remember(&result, HB_VDEC_Module_Uninit()); dec_module = 0; }
    if (enc_module) { remember(&result, HB_VENC_Module_Uninit()); enc_module = 0; }
    if (vp) { remember(&result, HB_VP_Exit()); vp = 0; }
    return result;
}
#define INIT_CHECK(call) do { int r = (call); if (r) { walle_close(); return r; } } while (0)
int walle_init(void) {
    if (vp || enc_module || dec_module) return -1;
    VP_CONFIG_S config = {0}; config.u32MaxPoolCnt = 32;
    INIT_CHECK(HB_VP_SetConfig(&config)); INIT_CHECK(HB_VP_Init()); vp = 1;
    INIT_CHECK(HB_VENC_Module_Init()); enc_module = 1;
    INIT_CHECK(HB_VDEC_Module_Init()); dec_module = 1;
    return 0;
}

int walle_decode(const uint8_t *jpeg, int length, uint8_t *out, int capacity, int width, int height) {
    if (!vp || !jpeg || !out || length < 4 || length > MAX_JPEG || width <= 0 || height <= 0
            || width > 1920 || height > 1080 || width % 2 || height % 2
            || capacity < width * height * 3 / 2) return -1;
    if (!dec_created) {
        VDEC_CHN_ATTR_S attr = {0};
        attr.enType = PT_JPEG; attr.enMode = VIDEO_MODE_FRAME;
        attr.enPixelFormat = HB_PIXEL_FORMAT_NV12;
        attr.u32FrameBufCnt = 3; attr.u32StreamBufCnt = 3;
        attr.u32StreamBufSize = MAX_JPEG;
        attr.bExternalBitStreamBuf = HB_FALSE;
        attr.stAttrJpeg.enMirrorFlip = DIRECTION_NONE;
        attr.stAttrJpeg.enRotation = CODEC_ROTATION_0;
        int r = HB_VDEC_CreateChn(DECODER, &attr); if (r) return r;
        dec_created = 1;
        r = HB_SYS_Alloc(&jpeg_physical, &jpeg_pixels, MAX_JPEG); if (r) return r;
        r = HB_VDEC_StartRecvStream(DECODER); if (r) return r;
        dec_started = 1;
    }
    memcpy(jpeg_pixels, jpeg, length);
    VIDEO_STREAM_S stream = {0};
    stream.pstPack.phy_ptr = jpeg_physical; stream.pstPack.vir_ptr = jpeg_pixels;
    stream.pstPack.size = length; stream.pstPack.src_idx = 0;
    int r = HB_VDEC_SendStream(DECODER, &stream, TIMEOUT_MS); if (r) return r;
    VIDEO_FRAME_S frame = {0};
    r = HB_VDEC_GetFrame(DECODER, &frame, TIMEOUT_MS); if (r) return r;
    int stride = frame.stVFrame.stride > 0 ? frame.stVFrame.stride : width;
    if (frame.stVFrame.width != (uint32_t)width || frame.stVFrame.height != (uint32_t)height
            || frame.stVFrame.pix_format != HB_PIXEL_FORMAT_NV12 || stride < width
            || !frame.stVFrame.vir_ptr[0] || !frame.stVFrame.vir_ptr[1]) {
        HB_VDEC_ReleaseFrame(DECODER, &frame); return -2;
    }
    for (int y = 0; y < height; ++y)
        memcpy(out + y * width, frame.stVFrame.vir_ptr[0] + y * stride, width);
    for (int y = 0; y < height / 2; ++y)
        memcpy(out + width * height + y * width, frame.stVFrame.vir_ptr[1] + y * stride, width);
    return HB_VDEC_ReleaseFrame(DECODER, &frame);
}

int walle_encoder_open(int width, int height, int fps, int kbps) {
    if (!vp || enc_created || width <= 0 || height <= 0 || width > 1920 || height > 1080
            || width % 2 || height % 2 || fps < 1 || fps > 30 || kbps < 50 || kbps > 3000) return -1;
    enc_width = width; enc_height = height;
    VENC_CHN_ATTR_S a = {0};
    a.stVencAttr.enType = PT_H264;
    a.stVencAttr.u32PicWidth = width; a.stVencAttr.u32PicHeight = height;
    a.stVencAttr.enPixelFormat = HB_PIXEL_FORMAT_NV12;
    a.stVencAttr.enMirrorFlip = VERTICAL;
    a.stVencAttr.enRotation = CODEC_ROTATION_0;
    a.stVencAttr.u32FrameBufferCount = 3; a.stVencAttr.u32BitStreamBufferCount = 3;
    a.stVencAttr.bExternalFreamBuffer = HB_TRUE;
    a.stVencAttr.u32BitStreamBufSize = (width * height * 3 / 2 + 1023) & ~1023;
    a.stVencAttr.vlc_buf_size = 2048 * 1024;
    a.stVencAttr.bEnableUserPts = HB_TRUE;
    a.stVencAttr.stAttrH264.h264_profile = HB_H264_PROFILE_BP;
    a.stVencAttr.stAttrH264.h264_level = HB_H264_LEVEL3_1;
    a.stRcAttr.enRcMode = VENC_RC_MODE_H264CBR;
    a.stRcAttr.stH264Cbr.u32FrameRate = fps;
    /* Periodic IDR bounds recovery for preencoded Packet tracks: aiortc's public
     * API does not expose PLI/REMB callbacks. Browser health still selects low.
     */
    a.stRcAttr.stH264Cbr.u32IntraPeriod = fps;
    a.stRcAttr.stH264Cbr.u32BitRate = kbps;
    a.stRcAttr.stH264Cbr.u32VbvBufferSize = 1000;
    a.stRcAttr.stH264Cbr.u32IntraQp = 30; a.stRcAttr.stH264Cbr.u32InitialRcQp = 63;
    a.stRcAttr.stH264Cbr.u32MinIQp = 8; a.stRcAttr.stH264Cbr.u32MaxIQp = 45;
    a.stRcAttr.stH264Cbr.u32MinPQp = 8; a.stRcAttr.stH264Cbr.u32MaxPQp = 45;
    a.stRcAttr.stH264Cbr.u32MinBQp = 8; a.stRcAttr.stH264Cbr.u32MaxBQp = 45;
    a.stRcAttr.stH264Cbr.bHvsQpEnable = HB_TRUE;
    a.stRcAttr.stH264Cbr.s32HvsQpScale = 2; a.stRcAttr.stH264Cbr.u32MaxDeltaQp = 10;
    a.stGopAttr.u32GopPresetIdx = 2; a.stGopAttr.s32DecodingRefreshType = 2;
    int r = HB_VENC_CreateChn(ENCODER, &a); if (r) return r;
    enc_created = 1;
    r = HB_VENC_SetChnAttr(ENCODER, &a); if (r) return r;
    r = HB_SYS_Alloc(&enc_physical, &enc_pixels, width * height * 3 / 2); if (r) return r;
    VENC_RECV_PIC_PARAM_S recv = {0};
    r = HB_VENC_StartRecvFrame(ENCODER, &recv); if (r) return r;
    enc_started = 1;
    return 0;
}
int walle_encode(const uint8_t *input, int length, uint8_t *out, int capacity, int64_t pts_ms) {
    if (!enc_started || !input || !out || length != enc_width * enc_height * 3 / 2) return -1;
    memcpy(enc_pixels, input, length);
    VIDEO_FRAME_S frame = {0};
    frame.stVFrame.width = enc_width; frame.stVFrame.height = enc_height;
    frame.stVFrame.size = length; frame.stVFrame.pix_format = HB_PIXEL_FORMAT_NV12;
    frame.stVFrame.phy_ptr[0] = enc_physical;
    frame.stVFrame.phy_ptr[1] = enc_physical + enc_width * enc_height;
    frame.stVFrame.vir_ptr[0] = enc_pixels;
    frame.stVFrame.vir_ptr[1] = (char *)enc_pixels + enc_width * enc_height;
    frame.stVFrame.pts = pts_ms;
    int r = HB_VENC_SendFrame(ENCODER, &frame, TIMEOUT_MS); if (r) return r;
    VIDEO_STREAM_S stream = {0};
    r = HB_VENC_GetStream(ENCODER, &stream, TIMEOUT_MS); if (r) return r;
    int size = stream.pstPack.size;
    if (size <= 0 || size > capacity || !stream.pstPack.vir_ptr) {
        HB_VENC_ReleaseStream(ENCODER, &stream); return -2;
    }
    memcpy(out, stream.pstPack.vir_ptr, size);
    r = HB_VENC_ReleaseStream(ENCODER, &stream);
    return r ? r : size;
}
