// Stable C ABI over the system WebRTC APM. All calls belong to one worker.
#include <webrtc/modules/audio_processing/include/audio_processing.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>

namespace {
constexpr int kRate = 48000;
constexpr int kSamples = 480;  // WebRTC requires exactly 10 ms.
struct Processor {
  std::unique_ptr<webrtc::AudioProcessing> apm;
  float gain;
};
}

extern "C" {
void* wali_apm_create(float pre_gain_db) {
  if (!std::isfinite(pre_gain_db) || pre_gain_db < -12 || pre_gain_db > 24)
    return nullptr;
  webrtc::Config config;
  config.Set<webrtc::ExtendedFilter>(new webrtc::ExtendedFilter(true));
  config.Set<webrtc::DelayAgnostic>(new webrtc::DelayAgnostic(true));
  auto p = std::unique_ptr<Processor>(new Processor);
  p->apm.reset(webrtc::AudioProcessing::Create(config));
  if (!p->apm) return nullptr;
  auto* apm = p->apm.get();
  // Moderate suppression preserves simultaneous near-end speech.
  if (apm->echo_cancellation()->set_suppression_level(
          webrtc::EchoCancellation::kModerateSuppression) ||
      apm->echo_cancellation()->Enable(true) ||
      apm->high_pass_filter()->Enable(true) ||
      apm->noise_suppression()->set_level(webrtc::NoiseSuppression::kModerate) ||
      apm->noise_suppression()->Enable(true) ||
      apm->gain_control()->set_mode(webrtc::GainControl::kFixedDigital) ||
      apm->gain_control()->set_target_level_dbfs(3) ||
      apm->gain_control()->enable_limiter(true) ||
      apm->gain_control()->Enable(true)) return nullptr;
  p->gain = std::pow(10.0f, pre_gain_db / 20.0f);
  return p.release();
}

void wali_apm_destroy(void* context) { delete static_cast<Processor*>(context); }

int wali_apm_render(void* context, const int16_t* pcm, int count) {
  if (!context || !pcm || count <= 0 || count % kSamples) return -1;
  auto* p = static_cast<Processor*>(context);
  const webrtc::StreamConfig stream(kRate, 1);
  for (int offset = 0; offset < count; offset += kSamples) {
    float samples[kSamples];
    for (int i = 0; i < kSamples; ++i) samples[i] = pcm[offset + i] / 32768.0f;
    const float* src[] = {samples};
    float* dst[] = {samples};
    int rc = p->apm->ProcessReverseStream(src, stream, stream, dst);
    if (rc) return rc;
  }
  return 0;
}

int wali_apm_capture(void* context, const int16_t* pcm, int16_t* output,
                     int count, int delay_ms) {
  if (!context || !pcm || !output || count <= 0 || count % kSamples ||
      delay_ms < 0 || delay_ms > 500) return -1;
  auto* p = static_cast<Processor*>(context);
  const webrtc::StreamConfig stream(kRate, 1);
  for (int offset = 0; offset < count; offset += kSamples) {
    float samples[kSamples];
    for (int i = 0; i < kSamples; ++i)
      samples[i] = std::max(-1.0f, std::min(1.0f, pcm[offset + i] / 32768.0f * p->gain));
    const float* src[] = {samples};
    float* dst[] = {samples};
    int rc = p->apm->set_stream_delay_ms(delay_ms);
    if (!rc) rc = p->apm->ProcessStream(src, stream, stream, dst);
    if (rc) return rc;
    for (int i = 0; i < kSamples; ++i)
      output[offset + i] = static_cast<int16_t>(std::max(-32768.0f,
          std::min(32767.0f, std::round(samples[i] * 32768.0f))));
  }
  return 0;
}
}
