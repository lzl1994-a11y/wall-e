#include <memory>
#include <mutex>
#include <string>

#include "btcpp_ros2_interfaces/action/execute_tree.hpp"
#include "nlohmann/json.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_msgs/msg/string.hpp"

using Json = nlohmann::json;

namespace {

constexpr char kTreeName[] = "WaliTask";
constexpr char kActionName[] = "wali_task";
constexpr char kLegacyExecuteTopic[] = "/behavior_tree/execute";
constexpr char kLegacyCancelTopic[] = "/behavior_tree/cancel";
constexpr char kLegacyStatusTopic[] = "/behavior_tree/status";

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
    execute_pub_ = create_publisher<std_msgs::msg::String>(kLegacyExecuteTopic, 10);
    cancel_pub_ = create_publisher<std_msgs::msg::String>(kLegacyCancelTopic, 10);
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
    }
    std_msgs::msg::String message;
    message.data = Json{{"plan_id", plan_id}}.dump();
    cancel_pub_->publish(message);
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle) {
    const auto goal = goal_handle->get_goal();
    const auto plan = Json::parse(goal->payload);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      goal_reserved_ = false;
      current_goal_ = goal_handle;
      plan_id_ = plan["plan_id"].get<std::string>();
    }
    if (execute_pub_->get_subscription_count() == 0) {
      finish_goal("failure", Json{{"plan_id", plan_id_},
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
      goal = current_goal_;
      value = status.value("status", "");
    }
    if (!is_terminal(value)) {
      auto feedback = std::make_shared<ExecuteTree::Feedback>();
      feedback->message = payload;
      goal->publish_feedback(feedback);
      return;
    }
    finish_goal(value, payload);
  }

  void finish_goal(const std::string& status, const std::string& payload) {
    std::shared_ptr<GoalHandle> goal;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      goal = current_goal_;
      current_goal_.reset();
      plan_id_.clear();
      goal_reserved_ = false;
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

  std::mutex state_mutex_;
  bool goal_reserved_ = false;
  std::string plan_id_;
  std::shared_ptr<GoalHandle> current_goal_;
  rclcpp_action::Server<ExecuteTree>::SharedPtr action_server_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr execute_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr cancel_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr status_sub_;
};

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WaliTaskActionBridge>());
  rclcpp::shutdown();
  return 0;
}
