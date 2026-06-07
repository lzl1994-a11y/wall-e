import os
import yaml
import tempfile
import wave
import dashscope
from dashscope.audio.asr import Transcription
import time

def test_transcription():
    print("=== 测试 Transcription.async_call 本地文件上传 ===")
    try:
        with open("core/config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        dashscope.api_key = config['ai_settings']['api_key']
    except Exception as e:
        print(f"读取配置失败: {e}")
        return

    import urllib.request
    tmp_path = "test_transcription.wav"
    url = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
    urllib.request.urlretrieve(url, tmp_path)
            
    # Windows 路径转 file:// URL，必须前缀 file:/// 并将反斜杠替换为正斜杠
    abs_path = os.path.abspath(tmp_path)
    file_url = "file://" + abs_path.replace("\\", "/")
    print(f"生成的本地文件 URL: {file_url}")

    try:
        print("调用 Transcription.async_call...")
        task_response = Transcription.async_call(
            model='sensevoice-v1',
            file_urls=[file_url]
        )
        
        print(f"任务已提交，Task ID: {task_response.output.task_id}")
        
        # 轮询等待任务完成
        while True:
            result = Transcription.wait(task=task_response.output.task_id)
            status = result.output.task_status
            if status == 'SUCCEEDED':
                print(f"任务成功！")
                print(f"结果: {result}")
                break
            elif status == 'FAILED':
                print(f"任务失败！错误: {result.output.message}")
                break
            else:
                print(f"当前状态: {status}，等待中...")
                time.sleep(1)
                
    except Exception as e:
        print(f"测试报错: {e}")
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    test_transcription()
