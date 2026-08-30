// Isolated C++ FC diagnostic: libretro + evdev + ALSA + WTFT.  It is not a robot node.
#include <alsa/asoundlib.h>
#include <arpa/inet.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <linux/input.h>
#include <netinet/in.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

struct GameInfo { const char* path; const void* data; size_t size; const char* meta; };
using Env = bool(*)(unsigned, void*); using Video = void(*)(const void*,unsigned,unsigned,size_t);
using Audio = void(*)(int16_t,int16_t); using AudioBatch = size_t(*)(const int16_t*,size_t);
using Poll = void(*)(); using Input = int16_t(*)(unsigned,unsigned,unsigned,unsigned);
constexpr unsigned ENV_PIXEL=10, PIXEL_XRGB8888=1, DEV_JOYPAD=1;
// libretro NES IDs: B=0, Select=2, Start=3, D-pad=4..7, A=8.
constexpr uint16_t A=1<<8,B=1<<0,SEL=1<<2,START=1<<3,UP=1<<4,DOWN=1<<5,LEFT=1<<6,RIGHT=1<<7;

class AudioOut {
 public:
  AudioOut(){ if(snd_pcm_open(&pcm_,"default",SND_PCM_STREAM_PLAYBACK,0)<0) throw std::runtime_error("cannot open ALSA default");
    if(snd_pcm_set_params(pcm_,SND_PCM_FORMAT_S16_LE,SND_PCM_ACCESS_RW_INTERLEAVED,2,48000,1,80000)<0) throw std::runtime_error("cannot configure ALSA"); worker_=std::thread(&AudioOut::run,this); }
  ~AudioOut(){ stop_=true; cv_.notify_all(); worker_.join(); snd_pcm_drop(pcm_); snd_pcm_close(pcm_); }
  void push(const int16_t* p,size_t frames){ std::lock_guard<std::mutex> l(mu_); for(size_t i=0;i<frames*2;i++){ int v=int(p[i])*2/5; q_.push_back(int16_t(v)); } while(q_.size()>19200) q_.pop_front(); cv_.notify_one(); }
 private:
  void run(){ std::vector<int16_t> out(960); while(!stop_){ { std::unique_lock<std::mutex> l(mu_); cv_.wait_for(l,std::chrono::milliseconds(10),[&]{return stop_||q_.size()>=960;}); for(auto& x:out){ if(q_.empty()) x=0; else{x=q_.front();q_.pop_front();} } }
    int n=snd_pcm_writei(pcm_,out.data(),480); if(n<0) snd_pcm_prepare(pcm_); } }
  snd_pcm_t* pcm_{}; std::deque<int16_t> q_; std::mutex mu_; std::condition_variable cv_; std::atomic<bool> stop_{false}; std::thread worker_;
};

class Pad {
 public:
  explicit Pad(const char* path){ fd_=open(path,O_RDONLY|O_NONBLOCK); if(fd_<0) throw std::runtime_error("cannot open controller"); int one=1; if(ioctl(fd_,EVIOCGRAB,one)<0) throw std::runtime_error("cannot grab controller"); th_=std::thread(&Pad::run,this); }
  ~Pad(){ stop_=true; th_.join(); int zero=0; ioctl(fd_,EVIOCGRAB,zero); close(fd_); }
  int16_t state(unsigned id) const { return (bits_.load()&(1u<<id))?1:0; }
 private:
  void set(uint16_t bit,bool down){ auto old=bits_.load(); while(!bits_.compare_exchange_weak(old,down?(old|bit):(old&~bit))); }
  void run(){ pollfd p{fd_,POLLIN,0}; while(!stop_){ if(poll(&p,1,100)<=0) continue; input_event e; while(read(fd_,&e,sizeof e)==sizeof e){ if(e.type==EV_KEY){ uint16_t m= e.code==304?A:e.code==305?B:e.code==307?A:e.code==308?B:e.code==314?SEL:e.code==315?START:0; if(m)set(m,e.value); } else if(e.type==EV_ABS&&(e.code==16||e.code==17)){ if(e.code==16){set(LEFT,e.value<0);set(RIGHT,e.value>0);}else{set(UP,e.value<0);set(DOWN,e.value>0);} } } } }
  int fd_{}; std::atomic<uint16_t> bits_{0}; std::atomic<bool> stop_{false}; std::thread th_;
};

class Tft {
 public:
  Tft(int fps):fps_(fps){ server_=socket(AF_INET,SOCK_STREAM,0); int one=1; setsockopt(server_,SOL_SOCKET,SO_REUSEADDR,&one,sizeof one); sockaddr_in a{};a.sin_family=AF_INET;a.sin_port=htons(9000);a.sin_addr.s_addr=htonl(INADDR_ANY); if(bind(server_,(sockaddr*)&a,sizeof a)||listen(server_,1))throw std::runtime_error("cannot listen WTFT"); std::cerr<<"waiting for TFT\n"; client_=accept(server_,nullptr,nullptr); if(client_<0)throw std::runtime_error("no TFT client"); uint8_t h[16]; recv_all(h,16); uint32_t n;memcpy(&n,h+12,4); n=ntohl(n);std::vector<uint8_t> body(n);if(n)recv_all(body.data(),n); sendmsg(0x10,1,{0xff,0xff,0xff,0xff,0,0,0,0,(uint8_t)(fps>>8),(uint8_t)fps,0,0}); }
  ~Tft(){ if(client_>=0){sendmsg(0x12,1,{});close(client_);}close(server_); }
  void frame(const uint8_t* data,unsigned w,unsigned h,size_t pitch){ auto now=std::chrono::steady_clock::now(); if(now-last_<std::chrono::milliseconds(1000/fps_))return; last_=now; cv::Mat bgra(h,w,CV_8UC4,const_cast<uint8_t*>(data),pitch),bgr,resized;cv::cvtColor(bgra,bgr,cv::COLOR_BGRA2BGR);double s=std::min(240.0/w,240.0/h);cv::resize(bgr,resized,cv::Size(int(w*s+.5),int(h*s+.5)),0,0,cv::INTER_AREA);std::vector<uint8_t> jpg;cv::imencode(".jpg",resized,jpg,{cv::IMWRITE_JPEG_QUALITY,70});sendmsg(0x11,(1u<<16)|frames_++,jpg); }
 private:
  void recv_all(uint8_t* p,size_t n){while(n){auto r=recv(client_,p,n,0);if(r<=0)throw std::runtime_error("TFT hello failed");p+=r;n-=r;}}
  void sendmsg(uint8_t type,uint32_t seq,const std::vector<uint8_t>& b){uint8_t h[16]={'W','T','F','T',1,type,0,0};uint32_t x=htonl(seq),n=htonl(b.size());memcpy(h+8,&x,4);memcpy(h+12,&n,4); if(send(client_,h,16,MSG_NOSIGNAL)!=16 || (!b.empty()&&send(client_,b.data(),b.size(),MSG_NOSIGNAL)!=(ssize_t)b.size()))throw std::runtime_error("TFT send failed");}
  int server_{-1},client_{-1},fps_,frames_{};std::chrono::steady_clock::time_point last_{};
};

struct Host { AudioOut* audio; Pad* pad; Tft* tft; }; static Host* g;
static bool env(unsigned c,void* p){if(c!=ENV_PIXEL)return false;return *(int*)p==PIXEL_XRGB8888;} static void vid(const void*p,unsigned w,unsigned h,size_t s){g->tft->frame((const uint8_t*)p,w,h,s);} static void aud(int16_t l,int16_t r){int16_t x[2]={l,r};g->audio->push(x,1);} static size_t audb(const int16_t*p,size_t n){g->audio->push(p,n);return n;} static int16_t input(unsigned port,unsigned dev,unsigned idx,unsigned id){return(port==0&&dev==DEV_JOYPAD&&idx==0)?g->pad->state(id):0;}
int main(int argc,char**argv){ if(argc<2){std::cerr<<"usage: "<<argv[0]<<" ROM [seconds] [fps]\n";return 2;} try { int seconds=argc>2?atoi(argv[2]):120,fps=argc>3?atoi(argv[3]):10; void* so=dlopen("/root/libretro-fceumm/fceumm_libretro.so",RTLD_NOW);if(!so)throw std::runtime_error(dlerror()); auto fn=[&](const char*n){return dlsym(so,n);}; auto setenv=(void(*)(Env))fn("retro_set_environment");auto setvid=(void(*)(Video))fn("retro_set_video_refresh");auto setaud=(void(*)(Audio))fn("retro_set_audio_sample");auto setaudb=(void(*)(AudioBatch))fn("retro_set_audio_sample_batch");auto setpoll=(void(*)(Poll))fn("retro_set_input_poll");auto setin=(void(*)(Input))fn("retro_set_input_state");auto init=(void(*)())fn("retro_init");auto load=(bool(*)(const GameInfo*))fn("retro_load_game");auto run=(void(*)())fn("retro_run");auto unload=(void(*)())fn("retro_unload_game");auto deinit=(void(*)())fn("retro_deinit"); AudioOut audio;Pad pad("/dev/input/event2");Tft tft(fps);Host host{&audio,&pad,&tft};g=&host;setenv(env);setvid(vid);setaud(aud);setaudb(audb);setpoll(+[]{});setin(input);init();GameInfo info{argv[1],nullptr,0,nullptr};if(!load(&info))throw std::runtime_error("ROM load failed");auto end=std::chrono::steady_clock::now()+std::chrono::seconds(seconds);while(std::chrono::steady_clock::now()<end){auto t=std::chrono::steady_clock::now();run();std::this_thread::sleep_until(t+std::chrono::microseconds(16667));}unload();deinit();dlclose(so);return 0;}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;} }
