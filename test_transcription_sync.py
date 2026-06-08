import os
import yaml
import dashscope

def test_transcription_sync():
    print("=== 测试 Transcription.call 同步上传 ===")
    try:
        with open("core/config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        dashscope.api_key = config['ai_settings']['api_key']
    except Exception as e:
        print(f"读取配置失败: {e}")
        return

    # 下载公开音频
    import urllib.request
    tmp_path = "test_transcription_sync.wav"
    url = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
    if not os.path.exists(tmp_path):
        urllib.request.urlretrieve(url, tmp_path)
            
    try:
        print("调用 Transcription.call...")
        # 尝试使用 dashscope.audio.asr.Transcription.call
        from dashscope.audio.asr import Transcription
        
        # 将本地文件转换为 file:// URL
        abs_path = os.path.abspath(tmp_path)
        file_url = "file://" + abs_path.replace("\\", "/")
        
        response = Transcription.call(
            model='sensevoice-v1',
            file_urls=[file_url]
        )
        print(f"结果: {response}")
    except Exception as e:
        print(f"测试报错: {e}")

if __name__ == "__main__":
    test_transcription_sync()
