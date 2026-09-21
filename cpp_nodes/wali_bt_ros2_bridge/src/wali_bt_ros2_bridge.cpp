#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "btcpp_ros2_interfaces/action/execute_tree.hpp"
#include "nlohmann/json.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/string.hpp"

using Json = nlohmann::json;
using namespace std::chrono_literals;

namespace {

constexpr char kTreeName[] = "WaliTask";
constexpr char kActionName[] = "wali_task";
constexpr char kLegacyExecuteTopic[] = "/behavior_tree/execute";
constexpr char kLegacyCancelTopic[] = "/behavior_tree/cancel";
constexpr char kLegacyStatusTopic[] = "/behavior_tree/status";
constexpr char kActionRequestTopic[] = "/action_request";

bool is_terminal(const std::string& status) {
  return status == "success" || status == "failure" || status == "halted" ||
         status == "rejected";
}

}  // namespace

class WaliTaskActionBridge : public rclcpp::Node {
 public:
  using ExecuteTree = btcpp_ros2_interfaces::action::ExecuteTree;
  using GoalHandle = rclcpp_action::ServerGoalHandle<ExecuteTree>;

  WaliTaskActionBridge() : Node("wali_task_action_bridge") {
    goal_timeout_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(std::max(
            1.0, declare_parameter<double>("goal_timeout_sec", 240.0))));
    cancel_grace_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(std::max(
            0.1, declare_parameter<double>("cancel_grace_sec", 2.0))));
    backend_loss_grace_ =
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(std::max(
                0.1,
                declare_parameter<double>("backend_loss_grace_sec", 2.0))));
    execute_pub_ = create_publisher<std_msgs::msg::String>(kLegacyExecuteTopic, 10);
    cancel_pub_ = create_publisher<std_msgs::msg::String>(kLegacyCancelTopic, 10);
    action_request_pub_ =
        create_publisher<std_msgs::msg::String>(kActionRequestTopic, 20);
    status_sub_ = create_subscription<std_msgs::msg::String>(
        kLegacyStatusTopic, 20,
        [this](std_msgs::msg::String::ConstSharedPtr message) {
          accept_legacy_status(message->data);
        });
    action_server_ = rclcpp_action::create_server<ExecuteTree>(
        this, kActionName,
        [this](const rclcpp_action::GoalUUID& uuid,
               std::shared_ptr<const ExecuteTree::Goal> goal) {
          return handle_goal(uuid, std::move(goal));
        },
        [this](const std::shared_ptr<GoalHandle> goal_handle) {
          return handle_cancel(goal_handle);
        },
        [this](const std::shared_ptr<GoalHandle> goal_handle) {
          handle_accepted(goal_handle);
        });
    watchdog_timer_ = create_wall_timer(250ms, [this]() { watchdog_tick(); });
    RCLCPP_INFO(get_logger(), "BehaviorTree.ROS2 ExecuteTree bridge is ready");
  }

 private:
  rclcpp_action::GoalResponse handle_goal(
      const rclcpp_action::GoalUUID&,
      const std::shared_ptr<const ExecuteTree::Goal> goal) {
    if (goal->target_tree != kTreeName || goal->payload.empty() ||
        goal->payload.size() > 65536) {
      return rclcpp_action::GoalResponse::REJECT;
    }
    try {
      const auto plan = Json::parse(goal->payload);
      if (!plan.is_object() || !plan.contains("plan_id") ||
          !plan["plan_id"].is_string() || plan["plan_id"].get<std::string>().empty()) {
        return rclcpp_action::GoalResponse::REJECT;
      }
    } catch (const std::exception&) {
      return rclcpp_action::GoalResponse::REJECT;
    }
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (goal_reserved_ || current_goal_) {
      return rclcpp_action::GoalResponse::REJECT;
    }
    goal_reserved_ = true;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
      const std::shared_ptr<GoalHandle> goal_handle) {
    std::string plan_id;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!current_goal_ || current_goal_ != goal_handle || plan_id_.empty()) {
        return rclcpp_action::CancelResponse::REJECT;
      }
      plan_id = plan_id_;
      cancel_requested_ = true;
      cancel_deadline_ = std::chrono::steady_clock::now() + cancel_grace_;
    }
    std_msgs::msg::String message;
    message.data = Json{{"plan_id", plan_id}}.dump();
    cancel_pub_->publish(message);
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle) {
    const auto goal = goal_handle->get_goal();
    const auto plan = Json::parse(goal->payload);
    const auto accepted_plan_id = plan["plan_id"].get<std::string>();
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      goal_reserved_ = false;
      current_goal_ = goal_handle;
      plan_id_ = accepted_plan_id;
      goal_deadline_ = std::chrono::steady_clock::now() + goal_timeout_;
      cancel_deadline_.reset();
      backend_missing_since_.reset();
      cancel_requested_ = false;
      recovery_in_progress_ = false;
    }
    if (execute_pub_->get_subscription_count() == 0) {
      finish_goal(accepted_plan_id, "failure",
                  Json{{"plan_id", accepted_plan_id},
                       {"status", "failure"},
                       {"results", Json::array()},
                       {"error", "legacy_behavior_tree_unavailable"}}
                      .dump());
      return;
    }
    std_msgs::msg::String message;
    message.data = goal->payload;
    execute_pub_->publish(message);
  }

  void accept_legacy_status(const std::string& payload) {
    Json status;
    try {
      status = Json::parse(payload);
    } catch (const std::exception&) {
      return;
    }
    std::shared_ptr<GoalHandle> goal;
    std::string value;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!current_goal_ || status.value("plan_id", "") != plan_id_) {
        return;
      }
      if (recovery_in_progress_) {
        return;
      }
      goal = current_goal_;
      value = status.value("status", "");
    }
    if (!is_terminal(value)) {
      auto feedback = std::make_shared<ExecuteTree::Feedback>();
      feedback->message = payload;
      goal->publish_feedback(feedback);
      return;
    }
    finish_goal(status.value("plan_id", ""), value, payload);
  }

  void watchdog_tick() {
    std::string expected_plan_id;
    std::string reason;
    bool requested_cancel = false;
    const auto now = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!current_goal_ || recovery_in_progress_) {
        backend_missing_since_.reset();
        return;
      }
      expected_plan_id = plan_id_;
      requested_cancel = cancel_requested_;
      if (execute_pub_->get_subscription_count() == 0) {
        if (!backend_missing_since_) {
          backend_missing_since_ = now;
        } else if (now - *backend_missing_since_ >= backend_loss_grace_) {
          reason = "legacy_behavior_tree_disconnected";
        }
      } else {
        backend_missing_since_.reset();
      }
      if (reason.empty() && cancel_deadline_ && now >= *cancel_deadline_) {
        reason = "legacy_behavior_tree_cancel_timeout";
      }
      if (reason.empty() && now >= goal_deadline_) {
        reason = "legacy_behavior_tree_goal_timeout";
      }
    }
    if (!reason.empty()) {
      recover_goal(expected_plan_id, reason, requested_cancel);
    }
  }

  void recover_goal(const std::string& expected_plan_id,
                    const std::string& reason, bool requested_cancel) {
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!current_goal_ || plan_id_ != expected_plan_id || recovery_in_progress_) {
        return;
      }
      recovery_in_progress_ = true;
    }

    std_msgs::msg::String cancel_message;
    cancel_message.data = Json{{"plan_id", expected_plan_id}}.dump();
    cancel_pub_->publish(cancel_message);

    std_msgs::msg::String stop_message;
    stop_message.data =
        Json{{"name", "stop_all"},
             {"arguments", Json::object()},
             {"request_id", recovery_request_id()},
             {"source", "wali_task_action_bridge_safety"},
             {"plan_id", expected_plan_id}}
            .dump();
    action_request_pub_->publish(stop_message);

    const auto terminal_status = requested_cancel ? "halted" : "failure";
    finish_goal(expected_plan_id, terminal_status,
                Json{{"plan_id", expected_plan_id},
                     {"status", terminal_status},
                     {"results", Json::array()},
                     {"error", reason},
                     {"source", "wali_task_action_bridge"}}
                    .dump());
  }

  void finish_goal(const std::string& expected_plan_id,
                   const std::string& status, const std::string& payload) {
    std::shared_ptr<GoalHandle> goal;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (!current_goal_ || plan_id_ != expected_plan_id) {
        return;
      }
      goal = current_goal_;
      current_goal_.reset();
      plan_id_.clear();
      goal_reserved_ = false;
      cancel_deadline_.reset();
      backend_missing_since_.reset();
      cancel_requested_ = false;
      recovery_in_progress_ = false;
    }
    if (!goal) {
      return;
    }
    auto result = std::make_shared<ExecuteTree::Result>();
    result->return_message = payload;
    result->node_status.status = status == "success"
                                     ? result->node_status.SUCCESS
                                     : result->node_status.FAILURE;
    if (status == "success") {
      goal->succeed(result);
    } else if (status == "halted" && goal->is_canceling()) {
      goal->canceled(result);
    } else {
      goal->abort(result);
    }
  }

  static std::string recovery_request_id() {
    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    return "bridge-recovery-" + std::to_string(ticks);
  }

  std::mutex state_mutex_;
  bool goal_reserved_ = false;
  bool cancel_requested_ = false;
  bool recovery_in_progress_ = false;
  std::string plan_id_;
  std::shared_ptr<GoalHandle> current_goal_;
  std::chrono::steady_clock::time_point goal_deadline_;
  std::optional<std::chrono::steady_clock::time_point> cancel_deadline_;
  std::optional<std::chrono::steady_clock::time_point> backend_missing_since_;
  std::chrono::steady_clock::duration goal_timeout_;
  std::chrono::steady_clock::duration cancel_grace_;
  std::chrono::steady_clock::duration backend_loss_grace_;
  rclcpp_action::Server<ExecuteTree>::SharedPtr action_server_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr execute_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr cancel_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr action_request_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr status_sub_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WaliTaskActionBridge>());
  rclcpp::shutdown();
  return 0;
}
