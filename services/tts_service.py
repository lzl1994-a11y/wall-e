# services/tts_service.py
import asyncio
import io
import edge_tts
import pygame
import threading
import queue
import time

class TTSService:
    """
    [ZH] 瓦力语音播报服务 (双线程极速流水线版)
         通过双队列分离网络 I/O (下载) 和本地阻塞 (播放)，实现无缝的“边播边下”。
    [EN] Wali Text-to-Speech Service (High-Speed Dual-Thread Pipeline Version)
         Separates network I/O (downloading) and local blocking (playback) using dual queues, 
         achieving seamless "download while playing".
    """
    
    def __init__(self, voice="zh-CN-YunxiaNeural"):
        self.voice = voice
        
        # [ZH] 文本队列：存放 Arbiter 扔进来的待播报文字
        # [EN] Text Queue: Stores pending text passed by the Arbiter
        self.text_queue = queue.Queue()   
        
        # [ZH] 音频队列：存放下载完成、准备在内存中播放的 MP3 二进制流
        # [EN] Audio Queue: Stores downloaded MP3 binary streams ready for in-memory playback
        self.audio_queue = queue.Queue()  
        
        # [ZH] 初始化 Pygame 音频混音器
        # [EN] Initialize Pygame audio mixer
        pygame.mixer.init()
        
        # [ZH] 线程 1：专职下载（网络 I/O 密集型）。设为守护线程(daemon=True)以便主进程退出时自动销毁。
        # [EN] Thread 1: Dedicated to downloading (Network I/O bound). Set as daemon to exit with main process.
        self.download_thread = threading.Thread(target=self._download_worker, daemon=True)
        self.download_thread.start()
        
        # [ZH] 线程 2：专职播放（本地阻塞型）。
        # [EN] Thread 2: Dedicated to playback (Local blocking).
        self.play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self.play_thread.start()
        
        print(f"[系统初始化] 🗣️ TTS 双线程服务已启动 / Dual-thread TTS Service started. Voice: {self.voice}")

    def speak(self, text):
        """
        [ZH] 外部调用接口。非阻塞：仅将文本放入队列后瞬间返回。
        [EN] External API. Non-blocking: Simply puts text into the queue and returns instantly.
        """
        if text and text.strip():
            self.text_queue.put(text.strip())

    def _download_worker(self):
        """
        [ZH] 下载工作线程：死盯文本队列，疯狂下载，绝不等待播放。
        [EN] Download worker thread: Monitors text queue, downloads aggressively, never waits for playback.
        """
        # [ZH] 为当前后台线程创建一个全新的异步事件循环
        # [EN] Create a brand new asynchronous event loop for the current background thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while True:
            # [ZH] 获取文本。如果队列为空，线程会在这里休眠等待，不消耗 CPU
            # [EN] Get text. If queue is empty, the thread sleeps here, consuming zero CPU
            text = self.text_queue.get()
            
            try:
                # [ZH] 定义内部异步函数来获取底层音频字节
                # [EN] Define internal async function to fetch raw audio bytes
                async def _get_bytes():
                    communicate = edge_tts.Communicate(text, self.voice,rate="+20%", pitch="+5Hz")
                    audio_data = b""
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_data += chunk["data"]
                    return audio_data

                # [ZH] 运行异步任务并拿到完整的二进制 MP3 数据
                # [EN] Run async task and get the complete binary MP3 data
                mp3_bytes = loop.run_until_complete(_get_bytes())
                
                # [ZH] 下载完成！立刻塞给播放线程，然后回头去接下一单
                # [EN] Download complete! Immediately push to playback thread, then loop back for the next task
                self.audio_queue.put(mp3_bytes)
                
            except Exception as e:
                print(f"❌ [TTS Download Error / 下载出错]: {e}")
            finally:
                # [ZH] 标记队列中的该任务已处理完成
                # [EN] Mark the task in the queue as done
                self.text_queue.task_done()

    def _play_worker(self):
        """
        [ZH] 播放工作线程：死盯音频队列，拿到二进制数据就播。
        [EN] Playback worker thread: Monitors audio queue, plays binary data as soon as it arrives.
        """
        while True:
            # [ZH] 获取已下载好的音频二进制流
            # [EN] Get the downloaded audio binary stream
            mp3_bytes = self.audio_queue.get()
            
            try:
                # [ZH] 将二进制流伪装成文件对象 (File-like object)，骗过 Pygame
                # [EN] Wrap binary stream into a file-like object to bypass Pygame's disk I/O requirement
                audio_stream = io.BytesIO(mp3_bytes)
                
                pygame.mixer.music.load(audio_stream)
                pygame.mixer.music.play()

                # [ZH] 核心机制：阻塞当前线程直到这句播完。
                #      注意：这里卡住不影响下载线程，下载线程还在隔壁干活！
                # [EN] Core mechanism: Block current thread until playback finishes.
                #      Note: Blocking here does NOT affect the download thread!
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05) 
                
                # [ZH] 播完后卸载内存，防止内存泄漏
                # [EN] Unload from memory after playing to prevent memory leaks
                pygame.mixer.music.unload()
                
            except Exception as e:
                print(f"❌ [TTS Playback Error / 播放出错]: {e}")
            finally:
                self.audio_queue.task_done()

# ==========================================
# [ZH] 独立测试模块
# [EN] Independent Testing Module
# ==========================================
if __name__ == "__main__":
    tts = TTSService()
    
    print(">>> [Test] Enqueueing sentences... / 开始塞入句子...")
    
    # [ZH] 瞬间塞入三句话，主程序不会被卡住
    # [EN] Instantly enqueue three sentences, main program won't be blocked
    tts.speak("我在说第一句话，这句有点长，需要播好几秒钟。")
    tts.speak("看！我在播第一句的时候，第二句话的音频其实已经默默躺在内存里了！")
    tts.speak("所以你现在听到的声音，绝对是无缝衔接的！")
    
    print(">>> [Test] Sentences enqueued. Main thread is free! / 句子塞入完毕，主程序已解放！")
    
    # [ZH] 防止主线程直接退出导致守护线程被杀
    # [EN] Prevent main thread from exiting, which would kill daemon threads
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Test] Exiting... / 测试退出...")