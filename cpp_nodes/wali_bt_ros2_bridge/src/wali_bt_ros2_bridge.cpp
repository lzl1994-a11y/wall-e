#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "behaviortree_cpp/action_node.h"
#include "behaviortree_ros2/tree_execution_server.hpp"
#include "nlohmann/json.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using Json = nlohmann::json;

namespace {

constexpr char kTreeName[] = "WaliTask";
constexpr char kLegacyExecuteTopic[] = "/behavior_tree/execute";
constexpr char kLegacyCancelTopic[] = "/behavior_tree/cancel";
constexpr char kLegacyStatusTopic[] = "/behavior_tree/status";

const char kBridgeTreeXml[] = R"(
<root BTCPP_format="4" main_tree_to_execute="WaliTask">
  <BehaviorTree ID="WaliTask">
    <LegacyTaskPlan/>
  </BehaviorTree>
</root>
)";

bool is_terminal(const std::string& status) {
  return status == "success" || status == "failure" || status == "halted" ||
         status == "rejected";
}

}  // namespace

class WaliTaskServer;

class LegacyTaskPlanNode : public BT::StatefulActionNode {
 public:
  LegacyTaskPlanNode(const std::string& name, const BT::NodeConfig& config,
                     WaliTaskServer* owner)
      : BT::StatefulActionNode(name, config), owner_(owner) {}

  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus onStart() override;
  BT::NodeStatus onRunning() override;
  void onHalted() override;

 private:
  WaliTaskServer* owner_;
};

class WaliTaskServer : public BT::TreeExecutionServer {
 public:
  explicit WaliTaskServer(const rclcpp::NodeOptions& options)
      : BT::TreeExecutionServer(
            std::make_shared<rclcpp::Node>("wali_task_action_server", options)) {
    execute_pub_ = node()->create_publisher<std_msgs::msg::String>(
        kLegacyExecuteTopic, 10);
    cancel_pub_ = node()->create_publisher<std_msgs::msg::String>(
        kLegacyCancelTopic, 10);
    status_sub_ = node()->create_subscription<std_msgs::msg::String>(
        kLegacyStatusTopic, 20,
        [this](std_msgs::msg::String::ConstSharedPtr message) {
          accept_legacy_status(message->data);
        });
  }

  bool start_legacy_plan() {
    const auto& payload = goalPayload();
    Json plan;
    try {
      plan = Json::parse(payload);
    } catch (const std::exception&) {
      return false;
    }
    const auto plan_id = plan.value("plan_id", "");
    if (plan_id.empty() || execute_pub_->get_subscription_count() == 0) {
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      plan_id_ = plan_id;
      terminal_status_.clear();
      terminal_payload_.clear();
      feedback_payload_.clear();
      feedback_dirty_ = false;
      cancel_sent_ = false;
    }
    std_msgs::msg::String message;
    message.data = payload;
    execute_pub_->publish(message);
    return true;
  }

  BT::NodeStatus poll_legacy_plan() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (terminal_status_.empty()) {
      return BT::NodeStatus::RUNNING;
    }
    return terminal_status_ == "success" ? BT::NodeStatus::SUCCESS
                                          : BT::NodeStatus::FAILURE;
  }

  void cancel_legacy_plan() {
    std::string plan_id;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (cancel_sent_ || plan_id_.empty() || !terminal_status_.empty()) {
        return;
      }
      cancel_sent_ = true;
      plan_id = plan_id_;
    }
    std_msgs::msg::String message;
    message.data = Json{{"plan_id", plan_id}}.dump();
    cancel_pub_->publish(message);
  }

 protected:
  bool onGoalReceived(const std::string& tree_name,
                      const std::string& payload) override {
    if (tree_name != kTreeName || payload.empty() || payload.size() > 65536) {
      return false;
    }
    try {
      const auto plan = Json::parse(payload);
      return plan.is_object() && plan.contains("plan_id") &&
             plan["plan_id"].is_string() && !plan["plan_id"].get<std::string>().empty();
    } catch (const std::exception&) {
      return false;
    }
  }

  void registerNodesIntoFactory(BT::BehaviorTreeFactory& factory) override {
    BT::NodeBuilder builder = [this](const std::string& name,
                                     const BT::NodeConfig& config) {
      return std::make_unique<LegacyTaskPlanNode>(name, config, this);
    };
    factory.registerBuilder<LegacyTaskPlanNode>("LegacyTaskPlan", builder);
    factory.registerBehaviorTreeFromText(kBridgeTreeXml);
  }

  std::optional<std::string> onLoopFeedback() override {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!feedback_dirty_) {
      return std::nullopt;
    }
    feedback_dirty_ = false;
    return feedback_payload_;
  }

  std::optional<std::string> onTreeExecutionCompleted(
      BT::NodeStatus status, bool was_cancelled) override {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!terminal_payload_.empty()) {
      return terminal_payload_;
    }
    return Json{{"status", was_cancelled ? "halted" : BT::toStr(status)}}.dump();
  }

 private:
  void accept_legacy_status(const std::string& payload) {
    Json status;
    try {
      status = Json::parse(payload);
    } catch (const std::exception&) {
      return;
    }
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (plan_id_.empty() || status.value("plan_id", "") != plan_id_) {
      return;
    }
    const auto value = status.value("status", "");
    feedback_payload_ = payload;
    feedback_dirty_ = true;
    if (is_terminal(value)) {
      terminal_status_ = value;
      terminal_payload_ = payload;
    }
  }

  std::mutex state_mutex_;
  std::string plan_id_;
  std::string terminal_status_;
  std::string terminal_payload_;
  std::string feedback_payload_;
  bool feedback_dirty_ = false;
  bool cancel_sent_ = false;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr execute_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr cancel_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr status_sub_;
};

BT::NodeStatus LegacyTaskPlanNode::onStart() {
  return owner_->start_legacy_plan() ? BT::NodeStatus::RUNNING
                                     : BT::NodeStatus::FAILURE;
}

BT::NodeStatus LegacyTaskPlanNode::onRunning() {
  return owner_->poll_legacy_plan();
}

void LegacyTaskPlanNode::onHalted() { owner_->cancel_legacy_plan(); }

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  auto server = std::make_shared<WaliTaskServer>(options);
  rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), 0, false, std::chrono::milliseconds(250));
  executor.add_node(server->node());
  executor.spin();
  executor.remove_node(server->node());
  rclcpp::shutdown();
  return 0;
}
