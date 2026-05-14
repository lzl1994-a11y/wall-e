# test_main.py
import sys
import json
import queue  # 新增：引入线程安全的队列模块
import re     # 🚀 新增：引入正则表达式模块，用于文本清洗

from services.llm_service import LLMService
from services.tts_service import TTSService
import services.mcp_service as mcp
def main():
    print("🚀 正在初始化大模型服务...")
    try:
        llm = LLMService()
        tts = TTSService()
    except Exception as e:
        print(f"初始化失败 (请检查 config.yaml): {e}")
        return

    print("✅ 瓦力系统就绪！(输入 'exit' 退出)")
    print("-" * 50)

    # 1. 创建一个 TTS 播报队列
    # 在未来的多线程架构中，你可以单开一个 TTS 线程，死循环从这个队列里 .get() 数据去播放
    tts_queue = tts.text_queue

    # 模拟多轮对话的历史记录
    chat_history = []
    
    # 2. 定义触发截断的标点符号集
    punctuations = {'。', '？', '.', '?'}

    while True:
        user_input = input("\n🗣️ [你对瓦力说]: ")
        if user_input.lower() == 'exit':
            break

        print("\n🔊 [瓦力的反应]:")
        
        # 实时打印用的缓存
        text_buffer = ""
        
        # 用于切分句子的缓存和计数器
        sentence_buffer = ""
        punc_count = 0  # 标点符号计数器
        
        # 调用流式接口
        stream = llm.chat_stream(user_input, chat_history)
        
        for data in stream:
            # 拿到文本数据
            if data["type"] == "text":
                # 注意：Kimi 有时候一个 chunk 会返回多个字符，所以我们遍历 chunk 里的字符
                chunk_text = data["content"]
                
                # 🚀 终端打印：原汁原味输出，包含表情包 😄👋，让屏幕显示保持生动
                sys.stdout.write(chunk_text)
                sys.stdout.flush()
                
                text_buffer += chunk_text
                
                # 逐个字符判断是否为标点
                for char in chunk_text:
                    sentence_buffer += char
                    if char in punctuations:
                        punc_count += 1
                        
                    # 当攒够 2 个标点符号时，打包送入队列
                    if punc_count >= 2:
                        clean_sentence = sentence_buffer.strip()
                        
                        # 🚀🚀🚀 核心修改：TTS 专属净化器！
                        # 剔除所有【不是】汉字、字母、数字、空白符和常用标点的字符 (消灭 Emoji)
                        tts_safe_sentence = re.sub(r'[^\w\s\u4e00-\u9fa5，。？！、：；“”（）《》.,?!]', '', clean_sentence)
                        
                        # 只有净化后还剩下真正的文字时，才送去播报
                        if tts_safe_sentence.strip():
                            tts_queue.put(tts_safe_sentence.strip())
                            print(f"\n   📥 [纯净文本入队]: {tts_safe_sentence.strip()}") 
                        
                        # 重置缓冲区和计数器，准备迎接下一个小长句
                        sentence_buffer = ""
                        punc_count = 0
                
            # 拿到完整拼装的工具数据：在真实系统中，这里丢给 MCP/硬件服务
            elif data["type"] == "tool_call":
                print(f"\n   ⚡ [准备执行动作]: 技能名称 = {data['name']}")
                
                try:
                    # 尝试调用 MCP 技能中枢
                    result = mcp.execute_tool(data['name'], data['arguments'])
                    print(f"   ✅ [动作执行成功]: {result}")
                except json.JSONDecodeError:
                    print(f"   ❌ [动作执行失败]: 大模型返回的参数格式错误 (缺括号等) -> {data['arguments']}")
                except Exception as e:
                    print(f"   ❌ [动作执行异常]: {e}")

        # 3. 扫尾工作：流式输出结束后，如果最后的一句话没凑够 2 个标点，也要把它塞进队列
        clean_tail = sentence_buffer.strip()
        if clean_tail:
            # 🚀 尾巴同样需要净化！
            tts_safe_tail = re.sub(r'[^\w\s\u4e00-\u9fa5，。？！、：；“”（）《》.,?!]', '', clean_tail)
            if tts_safe_tail.strip():
                tts_queue.put(tts_safe_tail.strip())
                print(f"\n   📥 [纯净尾巴入队]: {tts_safe_tail.strip()}")

        print("\n" + "-" * 50)
        
        # 保存对话历史
        chat_history.append({"role": "user", "content": user_input})
        if text_buffer:
             chat_history.append({"role": "assistant", "content": text_buffer})

if __name__ == "__main__":
    main()