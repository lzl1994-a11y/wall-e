#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

class Nv12PadderNode : public rclcpp::Node {
 public:
  Nv12PadderNode() : Node("nv12_padder_node") {
    input_topic_ = declare_parameter<std::string>("input_topic", "/image_nv12");
    output_topic_ = declare_parameter<std::string>("output_topic", "/image_padded_nv12");
    target_w_ = declare_parameter<int>("target_width", 960);
    target_h_ = declare_parameter<int>("target_height", 544);
    flip_vertical_ = declare_parameter<bool>("flip_vertical", false);
    flip_horizontal_ = declare_parameter<bool>("flip_horizontal", false);

    if (target_w_ <= 0 || target_h_ <= 0 || (target_w_ % 2) != 0 || (target_h_ % 2) != 0) {
      throw std::runtime_error("target_width/target_height must be positive and even for NV12");
    }

    reset_canvas();

    auto qos = rclcpp::QoS(rclcpp::KeepLast(2)).reliable();
    pub_ = create_publisher<sensor_msgs::msg::Image>(output_topic_, qos);
    sub_ = create_subscription<sensor_msgs::msg::Image>(
        input_topic_, qos,
        [this](sensor_msgs::msg::Image::ConstSharedPtr msg) { on_image(*msg); });

    RCLCPP_INFO(get_logger(), "Fast NV12 padder: %s -> %s, target=%dx%d, flip_v=%s, flip_h=%s",
                input_topic_.c_str(), output_topic_.c_str(), target_w_, target_h_,
                flip_vertical_ ? "true" : "false", flip_horizontal_ ? "true" : "false");
  }

 private:
  void reset_canvas() {
    const auto y_size = static_cast<size_t>(target_w_) * static_cast<size_t>(target_h_);
    const auto total_size = y_size * 3U / 2U;
    canvas_.assign(total_size, 0U);
    std::fill(canvas_.begin() + static_cast<std::ptrdiff_t>(y_size), canvas_.end(), 128U);
  }

  void on_image(const sensor_msgs::msg::Image& msg) {
    const int src_w = static_cast<int>(msg.width);
    const int src_h = static_cast<int>(msg.height);
    if (src_w <= 0 || src_h <= 0 || (src_w % 2) != 0 || (src_h % 2) != 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Invalid NV12 size: %dx%d", src_w, src_h);
      return;
    }
    if (src_w > target_w_ || src_h > target_h_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Input %dx%d exceeds target %dx%d", src_w, src_h, target_w_, target_h_);
      return;
    }

    const size_t expected = static_cast<size_t>(src_w) * static_cast<size_t>(src_h) * 3U / 2U;
    if (msg.data.size() < expected) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "NV12 data too short: got %zu, expected %zu", msg.data.size(), expected);
      return;
    }

    // Keep the padded border black even if the camera resolution changes at runtime.
    if (src_w != last_src_w_ || src_h != last_src_h_) {
      reset_canvas();
      last_src_w_ = src_w;
      last_src_h_ = src_h;
    }

    const int start_x = (target_w_ - src_w) / 2;
    const int start_y = (target_h_ - src_h) / 2;
    const auto* src = msg.data.data();
    auto* dst = canvas_.data();

    const size_t src_y_size = static_cast<size_t>(src_w) * static_cast<size_t>(src_h);
    const size_t dst_y_size = static_cast<size_t>(target_w_) * static_cast<size_t>(target_h_);

    for (int row = 0; row < src_h; ++row) {
      const int src_row_index = flip_vertical_ ? (src_h - 1 - row) : row;
      const auto* src_row = src + static_cast<size_t>(src_row_index) * static_cast<size_t>(src_w);
      auto* dst_row = dst + static_cast<size_t>(start_y + row) * static_cast<size_t>(target_w_) + static_cast<size_t>(start_x);
      
      if (flip_horizontal_) {
        for (int c = 0; c < src_w; ++c) {
          dst_row[c] = src_row[src_w - 1 - c];
        }
      } else {
        std::memcpy(dst_row, src_row, static_cast<size_t>(src_w));
      }
    }

    const auto* src_uv = src + src_y_size;
    auto* dst_uv = dst + dst_y_size;
    const int start_uv_y = start_y / 2;
    const int src_uv_h = src_h / 2;
    for (int row = 0; row < src_uv_h; ++row) {
      const int src_row_index = flip_vertical_ ? (src_uv_h - 1 - row) : row;
      const auto* src_row = src_uv + static_cast<size_t>(src_row_index) * static_cast<size_t>(src_w);
      auto* dst_row = dst_uv + static_cast<size_t>(start_uv_y + row) * static_cast<size_t>(target_w_) + static_cast<size_t>(start_x);
      
      if (flip_horizontal_) {
        for (int c = 0; c < src_w; c += 2) {
          dst_row[c] = src_row[src_w - 2 - c];
          dst_row[c + 1] = src_row[src_w - 1 - c];
        }
      } else {
        std::memcpy(dst_row, src_row, static_cast<size_t>(src_w));
      }
    }

    sensor_msgs::msg::Image out;
    out.header = msg.header;
    out.height = static_cast<uint32_t>(target_h_);
    out.width = static_cast<uint32_t>(target_w_);
    out.encoding = "nv12";
    out.is_bigendian = 0;
    out.step = static_cast<sensor_msgs::msg::Image::_step_type>(target_w_);
    out.data = canvas_;
    pub_->publish(std::move(out));
  }

  std::string input_topic_;
  std::string output_topic_;
  int target_w_ = 960;
  int target_h_ = 544;
  bool flip_vertical_ = false;
  bool flip_horizontal_ = false;
  int last_src_w_ = -1;
  int last_src_h_ = -1;
  std::vector<uint8_t> canvas_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<Nv12PadderNode>());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "nv12_padder_node failed: %s\n", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
