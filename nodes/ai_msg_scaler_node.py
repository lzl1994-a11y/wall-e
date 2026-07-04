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

        self.img_w = 640
        self.img_h = 480
        self.model_w = float(self.get_parameter("model_width").value)
        self.model_h = float(self.get_parameter("model_height").value)
        self.transform_mode = str(self.get_parameter("transform_mode").value)
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
            f"mode={self.transform_mode}, model={self.model_w}x{self.model_h}"
        )

    def raw_img_cb(self, msg):
        self.latest_raw_stamp = msg.header.stamp
        if msg.width > 0 and msg.height > 0:
            self.img_w = int(msg.width)
            self.img_h = int(msg.height)

    def _map_point(self, x, y):
        actual_w = float(self.img_w)
        actual_h = float(self.img_h)
        mode = self.transform_mode

        if mode == "none":
            return x, y

        if mode == "letterbox_fit":
            scale = min(self.model_w / actual_w, self.model_h / actual_h)
            pad_x = (self.model_w - actual_w * scale) / 2.0
            pad_y = (self.model_h - actual_h * scale) / 2.0
            return (x - pad_x) / scale, (y - pad_y) / scale

        if mode == "center_crop":
            scale = max(self.model_w / actual_w, self.model_h / actual_h)
            crop_x = (actual_w * scale - self.model_w) / 2.0
            crop_y = (actual_h * scale - self.model_h) / 2.0
            return (x + crop_x) / scale, (y + crop_y) / scale

        if mode == "physical_pad":
            pad_x = (self.model_w - actual_w) / 2.0
            pad_y = (self.model_h - actual_h) / 2.0
            return x - pad_x, y - pad_y

        # Default: detector stretches the camera image to the model input.
        return x * actual_w / self.model_w, y * actual_h / self.model_h

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
