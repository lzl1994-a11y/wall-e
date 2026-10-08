cmake_minimum_required(VERSION 3.16)
project(wali_webrtc_codec C)
add_library(wali_webrtc_codec SHARED src/codec.c)
target_include_directories(wali_webrtc_codec PRIVATE /usr/include/vio)
target_link_directories(wali_webrtc_codec PRIVATE /usr/lib/hbmedia)
target_compile_options(wali_webrtc_codec PRIVATE -O2 -Wall -Wextra -Werror)
# X3 SDK has circular runtime dependencies; load the same providers as hobot_codec.
target_link_options(wali_webrtc_codec PRIVATE -Wl,--no-as-needed -Wl,-z,defs)
target_link_libraries(wali_webrtc_codec PRIVATE hbmedia multimedia isp iar vio hbmem
    -l:libtinyalsa.so.2 dl pthread)
