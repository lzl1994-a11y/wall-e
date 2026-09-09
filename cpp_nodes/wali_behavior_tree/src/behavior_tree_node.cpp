#include <chrono>
#include <cstdio>
#include <cstdint>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "behaviortree_cpp/action_node.h"
#include "behaviortree_cpp/bt_factory.h"
#include "nlohmann/json.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;
using Json = nlohmann::json;

namespace {

constexpr char kExecuteTopic[] = "/behavior_tree/execute";
constexpr char kCancelTopic[] = "/behavior_tree/cancel";
constexpr char kTreeStatusTopic[] = "/behavior_tree/status";
constexpr char kActionCommandTopic[] = "/action_cmd";
constexpr char kActionStatusTopic[] = "/action_status";
constexpr std::size_t kMaxPlanSteps = 8;

const std::set<std::string> kAllowedActions = {
    "control_music", "express_emotion", "manual_servo", "move_chassis",
    "play_sequence", "set_tracking_mode", "set_vision_gate", "stop_all"};

const std::map<std::string, Json> kActionResources = {
    {"express_emotion", Json::array({"display"})},
    {"move_chassis", Json::array({"chassis"})},
    {"manual_servo", Json::array({"servo_motion"})},
    {"play_sequence", Json::array({"servo_motion"})},
    {"control_music", Json::array({"audio_music", "display"})},
    {"set_tracking_mode", Json::array({"camera", "chassis"})},
    {"set_vision_gate", Json::array({"camera"})},
    {"stop_all", Json::array({"servo_motion", "chassis", "audio_music", "camera"})},
};

const std::set<std::string> kTerminalActionStatuses = {
    "completed", "failed", "interrupted", "rejected"};

std::string random_id() {
  static std::mt19937_64 generator(std::random_device{}());
  static std::uniform_int_distribution<std::uint64_t> distribution;
  std::ostringstream stream;
  stream << std::hex << std::setfill('0') << std::setw(16) << distribution(generator)
         << std::setw(16) << distribution(generator);
  return stream.str();
}

struct PlanStep {
  std::string step_id;
  std::string name;
  Json arguments;
  Json original;
};

}  // namespace

class BehaviorTreeNode;

class RosActionNode : public BT::StatefulActionNode {
 public:
  RosActionNode(const std::string& name, const BT::NodeConfig& config,
                BehaviorTreeNode* owner)
      : BT::StatefulActionNode(name, config), owner_(owner) {}

  static BT::PortsList providedPorts() {
    return {BT::InputPort<unsigned>("step_index")};
  }

  BT::NodeStatus onStart() override;
  BT::NodeStatus onRunning() override;
  void onHalted() override;

 private:
  BehaviorTreeNode* owner_;
  unsigned step_index_ = 0;
};

class BehaviorTreeNode : public rclcpp::Node {
 public:
  BehaviorTreeNode() : Node("wali_behavior_tree_node") {
    action_timeout_ = std::chrono::duration<double>(
        declare_parameter<double>("action_timeout_sec", 20.0));

    action_command_pub_ = create_publisher<std_msgs::msg::String>(kActionCommandTopic, 10);
    tree_status_pub_ = create_publisher<std_msgs::msg::String>(kTreeStatusTopic, 10);
    execute_sub_ = create_subscription<std_msgs::msg::String>(
        kExecuteTopic, 10,
        [this](std_msgs::msg::String::ConstSharedPtr message) { on_plan(message->data); });
    cancel_sub_ = create_subscription<std_msgs::msg::String>(
        kCancelTopic, 10,
        [this](std_msgs::msg::String::ConstSharedPtr message) { on_cancel(message->data); });
    action_status_sub_ = create_subscription<std_msgs::msg::String>(
        kActionStatusTopic, 20,
        [this](std_msgs::msg::String::ConstSharedPtr message) {
          on_action_status(message->data);
        });

    BT::NodeBuilder builder = [this](const std::string& name,
                                     const BT::NodeConfig& config) {
      return std::make_unique<RosActionNode>(name, config, this);
    };
    factory_.registerBuilder<RosActionNode>("RosAction", builder);
    tick_timer_ = create_wall_timer(50ms, [this]() { tick_tree(); });
    RCLCPP_INFO(get_logger(), "Native BehaviorTree.CPP action-plan owner is ready");
  }

  bool start_step(unsigned index) {
    if (!tree_ || index >= steps_.size() || current_step_.has_value()) {
      return false;
    }
    current_step_ = index;
    current_request_id_ = random_id();
    current_terminal_.reset();
    step_deadline_ = std::chrono::steady_clock::now() +
                     std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                         action_timeout_);

    Json command = {
        {"name", steps_[index].name},
        {"arguments", steps_[index].arguments},
        {"request_id", current_request_id_},
        {"source", "native_behavior_tree"},
        {"plan_id", plan_id_},
        {"step_id", steps_[index].step_id},
    };
    publish(action_command_pub_, command);
    return true;
  }

  BT::NodeStatus poll_step(unsigned index) {
    if (!current_step_ || *current_step_ != index) {
      return BT::NodeStatus::FAILURE;
    }
    if (cancelled_) {
      record_current("interrupted", "plan_cancelled");
      publish_emergency_stop_once();
      return BT::NodeStatus::FAILURE;
    }
    if (std::chrono::steady_clock::now() >= step_deadline_) {
      record_current("timeout", "no_terminal_executor_status");
      publish_emergency_stop_once();
      return BT::NodeStatus::FAILURE;
    }
    if (!current_terminal_) {
      return BT::NodeStatus::RUNNING;
    }

    const auto terminal = *current_terminal_;
    const auto status = terminal.value("status", "failed");
    record_current(status, terminal.value("detail", ""),
                   terminal.value("source", "robot"));
    return status == "completed" ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
  }

  void halt_step(unsigned index) {
    if (current_step_ && *current_step_ == index) {
      record_current("interrupted", "tree_halted");
      publish_emergency_stop_once();
    }
  }

 private:
  friend class RosActionNode;

  bool validate_plan(const Json& plan, std::string& error) {
    if (!plan.is_object() || plan.value("schema_version", 0) != 1 ||
        plan.value("root_type", "") != "Sequence") {
      error = "unsupported_plan_schema";
      return false;
    }
    if (!plan.contains("plan_id") || !plan["plan_id"].is_string() ||
        plan["plan_id"].get<std::string>().empty() || !plan.contains("steps") ||
        !plan["steps"].is_array() || plan["steps"].empty() ||
        plan["steps"].size() > kMaxPlanSteps ||
        plan.value("on_failure", "") != "stop_remaining") {
      error = "invalid_plan_envelope";
      return false;
    }

    std::string previous;
    for (std::size_t index = 0; index < plan["steps"].size(); ++index) {
      const auto& step = plan["steps"][index];
      const auto expected_id = "step-" +
          (index + 1 < 10 ? std::string("0") : std::string()) +
          std::to_string(index + 1);
      if (!step.is_object() || step.value("step_id", "") != expected_id ||
          !step.contains("name") || !step["name"].is_string() ||
          kAllowedActions.count(step["name"].get<std::string>()) == 0 ||
          !step.contains("arguments") || !step["arguments"].is_object() ||
          !step.contains("depends_on") || !step["depends_on"].is_array() ||
          !step.contains("resources") || !step["resources"].is_array()) {
        error = "invalid_step_" + std::to_string(index + 1);
        return false;
      }
      const auto action_name = step["name"].get<std::string>();
      if (step["resources"] != kActionResources.at(action_name)) {
        error = "invalid_resources_" + std::to_string(index + 1);
        return false;
      }
      const auto expected_dependencies =
          index == 0 ? Json::array() : Json::array({previous});
      if (step["depends_on"] != expected_dependencies) {
        error = "invalid_dependency_" + std::to_string(index + 1);
        return false;
      }
      previous = expected_id;
    }
    return true;
  }

  void on_plan(const std::string& payload) {
    Json plan;
    std::string candidate_id;
    try {
      plan = Json::parse(payload);
      if (plan.contains("plan_id") && plan["plan_id"].is_string()) {
        candidate_id = plan["plan_id"].get<std::string>();
      }
    } catch (const std::exception& error) {
      publish_plan_status(candidate_id.empty() ? "invalid-plan" : candidate_id,
                          "rejected", {}, std::string("invalid_json:") + error.what());
      return;
    }

    std::string error;
    if (!validate_plan(plan, error)) {
      publish_plan_status(candidate_id.empty() ? "invalid-plan" : candidate_id,
                          "rejected", {}, error);
      return;
    }
    if (tree_) {
      publish_plan_status(candidate_id, "rejected", {}, "behavior_tree_busy");
      return;
    }

    plan_id_ = candidate_id;
    steps_.clear();
    results_ = Json::array();
    cancelled_ = false;
    stop_sent_ = false;
    for (const auto& item : plan["steps"]) {
      steps_.push_back({item["step_id"].get<std::string>(),
                        item["name"].get<std::string>(), item["arguments"], item});
    }

    std::ostringstream xml;
    xml << R"(<root BTCPP_format="4" main_tree_to_execute="MainTree">)"
        << R"(<BehaviorTree ID="MainTree"><Sequence name="ActionPlan">)";
    for (std::size_t index = 0; index < steps_.size(); ++index) {
      xml << R"(<RosAction step_index=")" << index << R"("/>)";
    }
    xml << "</Sequence></BehaviorTree></root>";

    try {
      tree_.emplace(factory_.createTreeFromText(xml.str()));
    } catch (const std::exception& exception) {
      publish_plan_status(plan_id_, "rejected", {},
                          std::string("tree_build_failed:") + exception.what());
      clear_plan();
      return;
    }
    publish_plan_status(plan_id_, "accepted", results_, "");
  }

  void on_cancel(const std::string& payload) {
    try {
      const auto request = Json::parse(payload);
      if (tree_ && request.value("plan_id", "") == plan_id_) {
        cancelled_ = true;
      }
    } catch (const std::exception&) {
      return;
    }
  }

  void on_action_status(const std::string& payload) {
    if (!current_step_) {
      return;
    }
    try {
      const auto status = Json::parse(payload);
      if (status.value("request_id", "") != current_request_id_ ||
          status.value("name", "") != steps_[*current_step_].name) {
        return;
      }
      const auto value = status.value("status", "");
      if (kTerminalActionStatuses.count(value) != 0) {
        current_terminal_ = status;
      }
    } catch (const std::exception&) {
      return;
    }
  }

  void tick_tree() {
    if (!tree_) {
      return;
    }
    const auto status = tree_->tickOnce();
    if (status == BT::NodeStatus::RUNNING) {
      return;
    }
    while (results_.size() < steps_.size()) {
      const auto& step = steps_[results_.size()];
      auto skipped = step.original;
      skipped["status"] = "skipped";
      skipped["action"] = step.name;
      skipped["reason"] = "prior_action_not_completed";
      skipped["node_status"] = "idle";
      results_.push_back(std::move(skipped));
    }
    const auto plan_status = cancelled_ ? "halted" :
        status == BT::NodeStatus::SUCCESS ? "success" : "failure";
    std::string error;
    if (plan_status != std::string("success")) {
      for (const auto& result : results_) {
        const auto item_status = result.value("status", "");
        if (item_status != "completed" && item_status != "skipped") {
          error = result.value("reason", item_status);
          break;
        }
      }
    }
    publish_plan_status(plan_id_, plan_status, results_, error);
    clear_plan();
  }

  void record_current(const std::string& status, const std::string& reason,
                      const std::string& executor = "") {
    if (!current_step_) {
      return;
    }
    auto result = steps_[*current_step_].original;
    result["status"] = status;
    result["action"] = steps_[*current_step_].name;
    result["request_id"] = current_request_id_;
    result["node_status"] = status == "completed" ? "success" :
        status == "interrupted" ? "halted" : "failure";
    if (!reason.empty()) {
      result["reason"] = reason;
    }
    if (!executor.empty()) {
      result["executor"] = executor;
    }
    results_.push_back(std::move(result));
    current_step_.reset();
    current_request_id_.clear();
    current_terminal_.reset();
  }

  void publish_emergency_stop_once() {
    if (stop_sent_) {
      return;
    }
    stop_sent_ = true;
    publish(action_command_pub_, Json{{"name", "stop_all"},
                                      {"arguments", Json::object()},
                                      {"request_id", random_id()},
                                      {"source", "native_behavior_tree_cancel"}});
  }

  void publish_plan_status(const std::string& plan_id, const std::string& status,
                           const Json& results, const std::string& error) {
    Json message = {{"plan_id", plan_id}, {"status", status},
                    {"results", results}, {"source", "native_behavior_tree"}};
    if (!error.empty()) {
      message["error"] = error;
    }
    publish(tree_status_pub_, message);
  }

  static void publish(const rclcpp::Publisher<std_msgs::msg::String>::SharedPtr& publisher,
                      const Json& payload) {
    std_msgs::msg::String message;
    message.data = payload.dump();
    publisher->publish(message);
  }

  void clear_plan() {
    tree_.reset();
    steps_.clear();
    results_ = Json::array();
    plan_id_.clear();
    current_step_.reset();
    current_request_id_.clear();
    current_terminal_.reset();
    cancelled_ = false;
    stop_sent_ = false;
  }

  BT::BehaviorTreeFactory factory_;
  std::optional<BT::Tree> tree_;
  std::vector<PlanStep> steps_;
  Json results_ = Json::array();
  std::string plan_id_;
  std::optional<unsigned> current_step_;
  std::string current_request_id_;
  std::optional<Json> current_terminal_;
  std::chrono::steady_clock::time_point step_deadline_;
  std::chrono::duration<double> action_timeout_{20.0};
  bool cancelled_ = false;
  bool stop_sent_ = false;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr action_command_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr tree_status_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr execute_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr cancel_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr action_status_sub_;
  rclcpp::TimerBase::SharedPtr tick_timer_;
};

BT::NodeStatus RosActionNode::onStart() {
  const auto input = getInput<unsigned>("step_index");
  if (!input) {
    return BT::NodeStatus::FAILURE;
  }
  step_index_ = input.value();
  return owner_->start_step(step_index_) ? BT::NodeStatus::RUNNING
                                         : BT::NodeStatus::FAILURE;
}

BT::NodeStatus RosActionNode::onRunning() { return owner_->poll_step(step_index_); }

void RosActionNode::onHalted() { owner_->halt_step(step_index_); }

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<BehaviorTreeNode>());
  } catch (const std::exception& error) {
    std::fprintf(stderr, "behavior_tree_node failed: %s\n", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
