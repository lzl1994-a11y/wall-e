#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import re

# 引入拼音转换库
from pypinyin import pinyin, Style

# 从你的 services/llm_service.py 中导入大模型底层驱动
from services.llm_service import LLMService

class LLMBrainNode(Node):
    def __init__(self):
        super().__init__('walle_llm_brain')
        
        # 1. 初始化底层大模型服务层
        try:
            self.llm = LLMService()
            self.get_logger().info('✅ 大模型服务层底层初始化成功')
        except Exception as e:
            self.get_logger().error(f'🔴 大模型服务层初始化失败: {e}')
            return

        self.chat_history = []
        self.punctuations = {'。', '？', '.', '?', '！', '!'}

        # 2. 订阅语音识别结果话题 (STT 节点 -> 本节点)
        self.voice_subscription = self.create_subscription(
            String,
            'voice_text',
            self.voice_callback,
            10
        )

        # 3. 声明发布者：发给语音播报话题 (本节点 -> TTS 节点)
        self.tts_publisher = self.create_publisher(String, 'tts_text', 10)
        
        # 4. 声明发布者：发给动作执行话题 (本节点 -> 动作/底盘/舵机节点)
        self.action_publisher = self.create_publisher(String, 'action_cmd', 10)

        # 🌟 5. 新增发布者：发布纠错后的完美文本话题 (本节点 -> 屏幕/UI/日志节点)
        self.corrected_publisher = self.create_publisher(String, 'corrected_text', 10)

        # 🌟 新增发布者：专门发给串口下位机的完整 AI 回复
        self.full_ai_publisher = self.create_publisher(String, 'full_ai_text', 10)

    def voice_callback(self, msg):
        """当语音识别节点（STT）监听到完整句子时，自动触发的大脑思考回调"""
        user_prompt = msg.data
        if not user_prompt:
            return
            
        self.get_logger().info(f'🧠 [大脑收到原始听觉]: "{user_prompt}"')
        
        # 1. 瞬间生成拼音对照表
        py_list = pinyin(user_prompt, style=Style.NORMAL)
        py_str = " ".join([item[0] for item in py_list])
        
        # 2. 🌟 升级版增强 Prompt：强行规范大模型的输出首行格式
        # 2. 🌟 究极增强版 Prompt：铁腕规范大模型的思考与输出格式
        augmented_prompt = (
            f"【原始语音识别文本】: \"{user_prompt}\"\n"
            f"【参考拼音对照表】: {py_str}\n\n"
            f"【核心指令与输出规范】（你必须严格遵守以下2条铁律，不可偏废）：\n"
            f"1. 纠错与首行拦截格式：你的回复的**第一行**，必须且只能是你结合拼音、上下文语境纠正后的标准文本。格式固定为：【修正文本】: [纠正后的文本]，随后**立刻紧跟一个换行符**（\\n）。绝对不能把这几个字漏掉，也不能在前面加任何前缀。\n"
            f"2. 意图响应：从**第二行**开始，根据你纠正后的真实意图，给出你作为一个实体机器人的正常对话回复，或者直接调用合适的动作工具。\n\n"
            f"请开始你的思考与响应："
        )
        
        self.get_logger().info(f'💡 [提示词已注入拼音与纠错规范]，正在请求大模型...')
        
        text_buffer = ""
        sentence_buffer = ""
        punc_count = 0
        
        # 🌟 状态控制变量：用于首行拦截
        corrected_text_extracted = False
        corrected_text = ""
        
        # 3. 传入加强版 prompt 让模型思考
        stream = self.llm.chat_stream(augmented_prompt, self.chat_history)
        
        for data in stream:
            # ==========================================
            # 语音通道：处理大模型生成的流式聊天文本
            # ==========================================
            if data["type"] == "text":
                chunk = data["content"]
                text_buffer += chunk
                sentence_buffer += chunk
                
                # 🌟 核心高能拦截算法：只在开头执行，捕获【修正文本】行
                if not corrected_text_extracted:
                    if "\n" in sentence_buffer:
                        parts = sentence_buffer.split("\n", 1)
                        first_line = parts[0].strip()
                        
                        if "【修正文本】:" in first_line:
                            # 提取出冒号后面的纯净正确文本
                            corrected_text = first_line.replace("【修正文本】:", "").strip()
                            
                            # 广播到 ROS 2 的 corrected_text 话题中
                            corr_msg = String()
                            corr_msg.data = corrected_text
                            self.corrected_publisher.publish(corr_msg)
                            self.get_logger().info(f'✨ [文本纠错成功] -> 原文: "{user_prompt}" | 修正后: "{corrected_text}"')
                            
                            # 🧠 极其关键：把第一行从缓冲区里无感剥离！剩下的才是真正要给 TTS 播报的对白
                            sentence_buffer = parts[1] if len(parts) > 1 else ""
                            corrected_text_extracted = True
                        else:
                            # 防御性逻辑：如果模型没听话首行没带标签，直接放行防止卡死
                            corrected_text_extracted = True
                    elif len(sentence_buffer) > 60:
                        # 防御性逻辑：攒了60个字还没换行，说明模型没用换行，强行放行处理
                        corrected_text_extracted = True
                    else:
                        # 换行符还没出来，继续攒着首行，不往下走 TTS 的标点切分，防止把控制台标签误播报
                        continue
                
                # --- 下面进入正常的 TTS 句子切分与净化分发流 ---
                for char in chunk:
                    # 如果当前字符还在第一行没处理完的碎碎片里，跳过
                    if not corrected_text_extracted:
                        break
                        
                    if char in self.punctuations:
                        punc_count += 1
                        
                    if punc_count >= 2:
                        clean_sentence = sentence_buffer.strip()
                        # 剔除 Emoji 等干扰项
                        tts_safe = re.sub(r'[^\w\s\u4e00-\u9fa5，。？！、：；“”（）《》.,?!]', '', clean_sentence)
                        
                        if tts_safe.strip():
                            out_msg = String()
                            out_msg.data = tts_safe.strip()
                            self.tts_publisher.publish(out_msg)
                            self.get_logger().info(f'📤 [分发单句语音]: {out_msg.data}')
                        
                        sentence_buffer = ""
                        punc_count = 0
                        
            # ==========================================
            # 动作通道：处理工具/动作调用
            # ==========================================
            elif data["type"] == "tool_call":
                self.get_logger().info(f'⚡ [决策出动作指令]: 技能名称 = {data["name"]}')
                action_msg = String()
                action_msg.data = json.dumps({
                    "name": data["name"],
                    "arguments": data["arguments"]
                })
                self.action_publisher.publish(action_msg)

        # 扫尾工作：流式输出结束后，处理末尾漏掉的文本
        clean_tail = sentence_buffer.strip()
        if clean_tail:
            tts_safe_tail = re.sub(r'[^\w\s\u4e00-\u9fa5，。？！、：；“”（）《》.,?!]', '', clean_tail)
            if tts_safe_tail.strip():
                out_msg = String()
                out_msg.data = tts_safe_tail.strip()
                self.tts_publisher.publish(out_msg)
                self.get_logger().info(f'📤 [分发尾部语音]: {out_msg.data}')

        # ==========================================
        # 🌟 记忆净化：用修正后的正确文本覆盖历史记录
        # ==========================================
        # 如果成功提取到了修正文本，记忆里就存绝对正确的文字；否则降级存原话

        final_user_memory = corrected_text if corrected_text else user_prompt
        self.chat_history.append({"role": "user", "content": final_user_memory})
        
        # 保存大模型的回答历史（需要过滤掉开头的【修正文本】行，保持历史干净）
        clean_assistant_memory = text_buffer
        if "【修正文本】:" in clean_assistant_memory and "\n" in clean_assistant_memory:
            clean_assistant_memory = clean_assistant_memory.split("\n", 1)[1]
            
        if clean_assistant_memory.strip():
            clean_text = clean_assistant_memory.strip()
            self.chat_history.append({"role": "assistant", "content": clean_text})
            
            # 👇 增加这 3 行代码：把完整的干净文本发布给串口节点
            full_msg = String()
            full_msg.data = clean_text
            self.full_ai_publisher.publish(full_msg)
        
       


def main(args=None):
    rclpy.init(args=args)
    node = LLMBrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()