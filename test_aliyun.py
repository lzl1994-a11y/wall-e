import os
import dashscope
import yaml

def test_api():
    print("=== 开始测试阿里云 API ===")
    try:
        with open("core/config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        api_key = config['ai_settings']['api_key']
        print(f"读取到的 API Key: {api_key[:8]}...{api_key[-4:]}")
    except Exception as e:
        print(f"读取 config.yaml 失败: {e}")
        return

    dashscope.api_key = api_key
    
    # 测试大模型文字生成权限
    print("\n1. 测试千问大模型 (文本) 权限...")
    try:
        from dashscope import Generation
        response = Generation.call(
            model='qwen-turbo',
            prompt='你好，请回复“连接正常”'
        )
        if response.status_code == 200:
            print(f"✅ 文字模型测试成功！回复: {response.output.text}")
        else:
            print(f"❌ 文字模型测试失败: {response.code} - {response.message}")
    except Exception as e:
        print(f"❌ 文字模型调用报错: {e}")

    # 测试语音识别权限 (尝试调用短音频文件识别)
    print("\n2. 测试语音识别 (SenseVoice/Paraformer) 权限...")
    try:
        import requests
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # 随便构造一个假的调用，看服务器是报 400(参数错误) 还是 403(无权限/未开通)
        payload = {
            "model": "sensevoice-v1",
            "input": {"url": "http://invalid-url.mp3"}
        }
        res = requests.post("https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription", json=payload, headers=headers)
        
        data = res.json()
        print(f"服务器原始返回: {data}")
        if "InvalidApiKey" in str(data) or "PermissionDenied" in str(data) or "NotEnable" in str(data) or "AccessDenied" in str(data):
            print("❌ 结论: 你的账号没有开通语音识别权限，或者 API Key 不正确。")
        else:
            print("✅ 语音识别服务状态看起来正常 (报错不是因为权限)。")

    except Exception as e:
        print(f"❌ 语音测试报错: {e}")

if __name__ == "__main__":
    test_api()
