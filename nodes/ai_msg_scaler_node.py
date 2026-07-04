#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

try:
    from ai_msgs.msg import PerceptionTargets
    from sensor_msgs.msg import Image
    HAS_MSGS = True
except ImportError:
    HAS_MSGS = False


class AIMSgScaler(Node):
    """Map detector boxes from model coordinates back to the camera image."""

    def __init__(self):
        super().__init__("ai_msg_scaler")
        if not HAS_MSGS:
            self.get_logger().error("Missing ai_msgs or sensor_msgs.")
            return

        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("ai_topic", "/hobot_mono2d_body_detection_raw")
        self.declare_parameter("output_topic", "/hobot_mono2d_body_detection")
        self.declare_parameter("model_width", 960.0)
        self.declare_parameter("model_height", 544.0)
        self.declare_parameter("transform_mode", "none")
        self.declare_parameter("x_scale", 1.0)
        self.declare_parameter("y_scale", 1.0)
        self.declare_parameter("x_offset", 0.0)
        self.declare_parameter("y_offset", 0.0)

        self.img_w = 640
        self.img_h = 480
        self.model_w = float(self.get_parameter("model_width").value)
        self.model_h = float(self.get_parameter("model_height").value)
        self.latest_raw_stamp = None

        image_topic = str(self.get_parameter("image_topic").value)
        ai_topic = str(self.get_parameter("ai_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)

        self.create_subscription(Image, image_topic, self.raw_img_cb, 10)
        self.create_subscription(PerceptionTargets, ai_topic, self.ai_cb, 10)
        self.pub = self.create_publisher(PerceptionTargets, output_topic, 10)

        self.get_logger().info(
            "AI box scaler started: "
            f"{ai_topic} -> {output_topic}, image={image_topic}, "
            f"mode={self._param_str('transform_mode')}, model={self.model_w}x{self.model_h}"
        )

    def raw_img_cb(self, msg):
        self.latest_raw_stamp = msg.header.stamp
        if msg.width > 0 and msg.height > 0:
            self.img_w = int(msg.width)
            self.img_h = int(msg.height)

    def _map_point(self, x, y):
        actual_w = float(self.img_w)
        actual_h = float(self.img_h)
        mode = self._param_str("transform_mode")

        if mode == "none":
            mapped_x, mapped_y = x, y

        elif mode == "letterbox_fit":
            scale = min(self.model_w / actual_w, self.model_h / actual_h)
            pad_x = (self.model_w - actual_w * scale) / 2.0
            pad_y = (self.model_h - actual_h * scale) / 2.0
            mapped_x, mapped_y = (x - pad_x) / scale, (y - pad_y) / scale

        elif mode == "center_crop":
            scale = max(self.model_w / actual_w, self.model_h / actual_h)
            crop_x = (actual_w * scale - self.model_w) / 2.0
            crop_y = (actual_h * scale - self.model_h) / 2.0
            mapped_x, mapped_y = (x + crop_x) / scale, (y + crop_y) / scale

        elif mode == "physical_pad":
            pad_x = (self.model_w - actual_w) / 2.0
            pad_y = (self.model_h - actual_h) / 2.0
            mapped_x, mapped_y = x - pad_x, y - pad_y

        else:
            # Detector stretches the camera image to the model input.
            mapped_x = x * actual_w / self.model_w
            mapped_y = y * actual_h / self.model_h

        mapped_x = mapped_x * self._param_float("x_scale", 1.0) + self._param_float("x_offset", 0.0)
        mapped_y = mapped_y * self._param_float("y_scale", 1.0) + self._param_float("y_offset", 0.0)
        return mapped_x, mapped_y

    def _param_float(self, name, default):
        try:
            return float(self.get_parameter(name).value)
        except Exception:
            return default

    def _param_str(self, name):
        try:
            return str(self.get_parameter(name).value)
        except Exception:
            return ""

    def _clip_rect(self, x1, y1, x2, y2):
        x1 = max(0.0, min(float(self.img_w), x1))
        y1 = max(0.0, min(float(self.img_h), y1))
        x2 = max(0.0, min(float(self.img_w), x2))
        y2 = max(0.0, min(float(self.img_h), y2))

        left = int(round(min(x1, x2)))
        top = int(round(min(y1, y2)))
        right = int(round(max(x1, x2)))
        bottom = int(round(max(y1, y2)))
        return left, top, max(0, right - left), max(0, bottom - top)

    def ai_cb(self, msg):
        for target in msg.targets:
            for roi in target.rois:
                rect = roi.rect
                x1, y1 = self._map_point(
                    float(rect.x_offset),
                    float(rect.y_offset),
                )
                x2, y2 = self._map_point(
                    float(rect.x_offset + rect.width),
                    float(rect.y_offset + rect.height),
                )
                new_x, new_y, new_w, new_h = self._clip_rect(x1, y1, x2, y2)

                rect.x_offset = new_x
                rect.y_offset = new_y
                rect.width = new_w
                rect.height = new_h

        if self.latest_raw_stamp is not None:
            msg.header.stamp = self.latest_raw_stamp

        self.pub.publish(msg)


def main():
    if not HAS_MSGS:
        return
    rclpy.init()
    node = AIMSgScaler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
