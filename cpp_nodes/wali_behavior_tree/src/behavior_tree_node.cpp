#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
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
constexpr char kActionRequestTopic[] = "/action_request";
constexpr char kActionStatusTopic[] = "/action_status";
constexpr char kVisualSearchRequestTopic[] = "/visual_search/request";
constexpr char kVisualSearchStatusTopic[] = "/visual_search/status";
constexpr std::size_t kMaxPlanSteps = 8;

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
  unsigned timeout_ms;
  unsigned max_attempts;
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

class RecoveryStopNode : public BT::SyncActionNode {
 public:
  RecoveryStopNode(const std::string& name, const BT::NodeConfig& config,
                   BehaviorTreeNode* owner)
      : BT::SyncActionNode(name, config), owner_(owner) {}

  static BT::PortsList providedPorts() {
    return {BT::InputPort<unsigned>("step_index")};
  }

  BT::NodeStatus tick() override;

 private:
  BehaviorTreeNode* owner_;
};

class VisualSearchNode : public BT::StatefulActionNode {
 public:
  VisualSearchNode(const std::string& name, const BT::NodeConfig& config,
                   BehaviorTreeNode* owner)
      : BT::StatefulActionNode(name, config), owner_(owner) {}
  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus onStart() override;
  BT::NodeStatus onRunning() override;
  void onHalted() override;

 private:
  BehaviorTreeNode* owner_;
};

class SearchNotFoundNode : public BT::SyncActionNode {
 public:
  SearchNotFoundNode(const std::string& name, const BT::NodeConfig& config,
                     BehaviorTreeNode* owner)
      : BT::SyncActionNode(name, config), owner_(owner) {}
  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus tick() override;

 private:
  BehaviorTreeNode* owner_;
};

class BehaviorTreeNode : public rclcpp::Node {
 public:
  BehaviorTreeNode() : Node("wali_behavior_tree_node") {
    load_skill_registry(declare_parameter<std::string>(
        "skill_registry_path", "core/action_skills.json"));
    action_timeout_ = std::chrono::duration<double>(
        declare_parameter<double>("action_timeout_sec", 20.0));
    visual_search_tree_path_ = declare_parameter<std::string>(
        "visual_search_tree_path", "core/behavior_trees/visual_search.xml");

    action_command_pub_ = create_publisher<std_msgs::msg::String>(kActionRequestTopic, 10);
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
    BT::NodeBuilder recovery_builder = [this](const std::string& name,
                                              const BT::NodeConfig& config) {
      return std::make_unique<RecoveryStopNode>(name, config, this);
    };
    factory_.registerBuilder<RecoveryStopNode>("RecoveryStop", recovery_builder);
    BT::NodeBuilder visual_builder = [this](const std::string& name,
                                            const BT::NodeConfig& config) {
      return std::make_unique<VisualSearchNode>(name, config, this);
    };
    factory_.registerBuilder<VisualSearchNode>("DetectVisualTarget", visual_builder);
    BT::NodeBuilder not_found_builder = [this](const std::string& name,
                                               const BT::NodeConfig& config) {
      return std::make_unique<SearchNotFoundNode>(name, config, this);
    };
    factory_.registerBuilder<SearchNotFoundNode>("ReportTargetNotFound",
                                                  not_found_builder);
    visual_search_request_pub_ =
        create_publisher<std_msgs::msg::String>(kVisualSearchRequestTopic, 10);
    visual_search_status_sub_ = create_subscription<std_msgs::msg::String>(
        kVisualSearchStatusTopic, 10,
        [this](std_msgs::msg::String::ConstSharedPtr message) {
          on_visual_search_status(message->data);
        });
    tick_timer_ = create_wall_timer(50ms, [this]() { tick_tree(); });
    RCLCPP_INFO(get_logger(), "Native BehaviorTree.CPP action-plan owner is ready");
  }

  bool start_step(unsigned index) {
    if (!tree_ || cancelled_ || index >= steps_.size() ||
        current_step_.has_value() ||
        attempt_counts_[index] >= steps_[index].max_attempts) {
      return false;
    }
    current_step_ = index;
    current_request_id_ = random_id();
    current_terminal_.reset();
    ++attempt_counts_[index];
    const auto configured_timeout = std::chrono::milliseconds(steps_[index].timeout_ms);
    const auto timeout_cap = std::chrono::duration_cast<std::chrono::milliseconds>(
        action_timeout_);
    step_deadline_ = std::chrono::steady_clock::now() +
                     std::min(configured_timeout, timeout_cap);

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
      return BT::NodeStatus::FAILURE;
    }
    if (!current_terminal_) {
      return BT::NodeStatus::RUNNING;
    }

    const auto terminal = *current_terminal_;
    const auto status = terminal.value("status", "failed");
    if (status == "rejected" || status == "interrupted") {
      attempt_counts_[index] = steps_[index].max_attempts;
    }
    record_current(status, terminal.value("detail", ""),
                   terminal.value("source", "robot"));
    return status == "completed" ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
  }

  void halt_step(unsigned index) {
    if (current_step_ && *current_step_ == index) {
      record_current("interrupted", "tree_halted");
    }
  }

  void recover_step(unsigned index) {
    if (index >= steps_.size() || index >= attempt_results_.size()) {
      return;
    }
    auto result = steps_[index].original;
    if (!attempt_results_[index].empty()) {
      result = attempt_results_[index].back();
    } else {
      result["status"] = "failed";
      result["action"] = steps_[index].name;
      result["reason"] = "action_retry_exhausted";
      result["node_status"] = "failure";
    }
    result["attempts"] = attempt_results_[index];
    results_.push_back(std::move(result));
    publish_emergency_stop_once();
  }

  bool start_visual_search() {
    if (!tree_ || cancelled_ || !visual_request_id_.empty() ||
        visual_attempt_ >= visual_max_views_) {
      return false;
    }
    ++visual_attempt_;
    visual_request_id_ = random_id();
    visual_terminal_.reset();
    visual_deadline_ = std::chrono::steady_clock::now() + 45s;
    publish(visual_search_request_pub_, Json{
        {"request_id", visual_request_id_}, {"plan_id", plan_id_},
        {"target", visual_target_}, {"question", visual_question_},
        {"attempt", visual_attempt_}, {"max_views", visual_max_views_}});
    return true;
  }

  BT::NodeStatus poll_visual_search() {
    if (cancelled_) {
      visual_request_id_.clear();
      visual_terminal_.reset();
      return BT::NodeStatus::FAILURE;
    }
    if (std::chrono::steady_clock::now() >= visual_deadline_) {
      last_visual_status_ = "uncertain";
      last_visual_evidence_ = "visual_search_timeout";
      last_visual_response_ = "这次画面分析超时了。";
      visual_request_id_.clear();
      visual_terminal_.reset();
      return BT::NodeStatus::FAILURE;
    }
    if (!visual_terminal_) {
      return BT::NodeStatus::RUNNING;
    }
    const auto result = *visual_terminal_;
    last_visual_status_ = result.value("status", "uncertain");
    last_visual_evidence_ = result.value("evidence", "");
    last_visual_response_ = result.value("response", "");
    visual_request_id_.clear();
    visual_terminal_.reset();
    if (last_visual_status_ == "found") {
      record_visual_result(true);
      return BT::NodeStatus::SUCCESS;
    }
    return BT::NodeStatus::FAILURE;
  }

  void halt_visual_search() {
    visual_request_id_.clear();
    visual_terminal_.reset();
  }

  void report_target_not_found() { record_visual_result(false); }

 private:
  friend class RosActionNode;

  void load_skill_registry(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
      throw std::runtime_error("cannot open action skill registry: " + path);
    }
    Json registry;
    input >> registry;
    if (!registry.is_object() || registry.value("schema_version", 0) != 1 ||
        !registry.contains("skills") || !registry["skills"].is_object()) {
      throw std::runtime_error("invalid action skill registry schema");
    }
    for (auto item = registry["skills"].begin();
         item != registry["skills"].end(); ++item) {
      const auto& definition = item.value();
      if (!definition.is_object() || !definition.contains("plan_resources") ||
          !definition["plan_resources"].is_array() ||
          definition["plan_resources"].empty() ||
          !definition.contains("action_bus") ||
          !definition["action_bus"].is_boolean()) {
        throw std::runtime_error("invalid action skill definition: " + item.key());
      }
      if (!definition["action_bus"].get<bool>()) {
        continue;
      }
      if (!definition.contains("timeout_ms") ||
          !definition["timeout_ms"].is_number_unsigned() ||
          definition["timeout_ms"].get<unsigned>() < 100 ||
          definition["timeout_ms"].get<unsigned>() > 60000 ||
          !definition.contains("max_attempts") ||
          !definition["max_attempts"].is_number_unsigned() ||
          definition["max_attempts"].get<unsigned>() < 1 ||
          definition["max_attempts"].get<unsigned>() > 3) {
        throw std::runtime_error("invalid action retry policy: " + item.key());
      }
      action_skills_[item.key()] = definition;
    }
    if (action_skills_.empty()) {
      throw std::runtime_error("action skill registry is empty");
    }
  }

  bool validate_plan(const Json& plan, std::string& error) {
    if (!plan.is_object()) {
      error = "unsupported_plan_schema";
      return false;
    }
    const auto root_type = plan.value("root_type", "");
    const bool visual_search = root_type == "VisualSearch";
    if ((!visual_search &&
         (plan.value("schema_version", 0) != 2 || root_type != "Sequence")) ||
        (visual_search && plan.value("schema_version", 0) != 3)) {
      error = "unsupported_plan_schema";
      return false;
    }
    if (!plan.contains("plan_id") || !plan["plan_id"].is_string() ||
        plan["plan_id"].get<std::string>().empty() || !plan.contains("steps") ||
        !plan["steps"].is_array() || plan["steps"].empty() ||
        plan["steps"].size() > kMaxPlanSteps ||
        plan.value("on_failure", "") !=
            (visual_search ? "report_not_found" : "stop_remaining")) {
      error = "invalid_plan_envelope";
      return false;
    }
    if (visual_search) {
      if (plan["steps"].size() != 1 || !plan.contains("target") ||
          !plan["target"].is_string() || plan["target"].get<std::string>().empty() ||
          plan["target"].get<std::string>().size() > 200 ||
          !plan.contains("question") || !plan["question"].is_string() ||
          plan["question"].get<std::string>().empty() ||
          plan["question"].get<std::string>().size() > 500 ||
          !plan.contains("max_views") || !plan["max_views"].is_number_unsigned() ||
          plan["max_views"].get<unsigned>() < 2 ||
          plan["max_views"].get<unsigned>() > 4) {
        error = "invalid_visual_search_plan";
        return false;
      }
    }

    std::string previous;
    for (std::size_t index = 0; index < plan["steps"].size(); ++index) {
      const auto& step = plan["steps"][index];
      const auto expected_id = "step-" +
          (index + 1 < 10 ? std::string("0") : std::string()) +
          std::to_string(index + 1);
      if (!step.is_object() || step.value("step_id", "") != expected_id ||
          !step.contains("name") || !step["name"].is_string() ||
          !step.contains("arguments") || !step["arguments"].is_object() ||
          !step.contains("depends_on") || !step["depends_on"].is_array() ||
          !step.contains("resources") || !step["resources"].is_array() ||
          !step.contains("timeout_ms") || !step["timeout_ms"].is_number_unsigned() ||
          !step.contains("max_attempts") ||
          !step["max_attempts"].is_number_unsigned()) {
        error = "invalid_step_" + std::to_string(index + 1);
        return false;
      }
      const auto action_name = step["name"].get<std::string>();
      const auto skill = action_skills_.find(action_name);
      if (skill == action_skills_.end()) {
        error = "unknown_action_" + std::to_string(index + 1);
        return false;
      }
      if (step["resources"] != skill->second["plan_resources"]) {
        error = "invalid_resources_" + std::to_string(index + 1);
        return false;
      }
      const auto expected_attempts = visual_search
          ? plan["max_views"].get<unsigned>() - 1
          : skill->second["max_attempts"].get<unsigned>();
      if (step["timeout_ms"] != skill->second["timeout_ms"] ||
          step["max_attempts"].get<unsigned>() != expected_attempts ||
          (visual_search && action_name != "move_chassis")) {
        error = "invalid_execution_policy_" + std::to_string(index + 1);
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
                          "rejected", Json::array(),
                          std::string("invalid_json:") + error.what());
      return;
    }

    std::string error;
    if (!validate_plan(plan, error)) {
      publish_plan_status(candidate_id.empty() ? "invalid-plan" : candidate_id,
                          "rejected", Json::array(), error);
      return;
    }
    if (tree_) {
      publish_plan_status(candidate_id, "rejected", Json::array(),
                          "behavior_tree_busy");
      return;
    }

    plan_id_ = candidate_id;
    root_type_ = plan.value("root_type", "Sequence");
    if (root_type_ == "VisualSearch") {
      visual_target_ = plan["target"].get<std::string>();
      visual_question_ = plan["question"].get<std::string>();
      visual_max_views_ = plan["max_views"].get<unsigned>();
    }
    steps_.clear();
    results_ = Json::array();
    attempt_results_.clear();
    attempt_counts_.clear();
    cancelled_ = false;
    stop_sent_ = false;
    for (const auto& item : plan["steps"]) {
      steps_.push_back({item["step_id"].get<std::string>(),
                        item["name"].get<std::string>(), item["arguments"],
                        item["timeout_ms"].get<unsigned>(),
                        item["max_attempts"].get<unsigned>(), item});
    }
    attempt_results_.assign(steps_.size(), Json::array());
    attempt_counts_.assign(steps_.size(), 0);

    try {
      if (root_type_ == "VisualSearch") {
        auto blackboard = BT::Blackboard::create();
        blackboard->set<int>("search_retries",
                             static_cast<int>(visual_max_views_ - 1));
        tree_.emplace(factory_.createTreeFromFile(
            visual_search_tree_path_, blackboard));
      } else {
        std::ostringstream xml;
        xml << R"(<root BTCPP_format="4" main_tree_to_execute="MainTree">)"
            << R"(<BehaviorTree ID="MainTree"><Sequence name="ActionPlan">)";
        for (std::size_t index = 0; index < steps_.size(); ++index) {
          xml << R"(<Fallback name="StepRecovery">)"
              << R"(<RetryUntilSuccessful num_attempts=")"
              << steps_[index].max_attempts << R"(">)"
              << R"(<Timeout msec=")" << steps_[index].timeout_ms + 100
              << R"("><RosAction step_index=")" << index
              << R"("/></Timeout></RetryUntilSuccessful>)"
              << R"(<RecoveryStop step_index=")" << index
              << R"("/></Fallback>)";
        }
        xml << "</Sequence></BehaviorTree></root>";
        tree_.emplace(factory_.createTreeFromText(xml.str()));
      }
    } catch (const std::exception& exception) {
      publish_plan_status(plan_id_, "rejected", Json::array(),
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

  void on_visual_search_status(const std::string& payload) {
    if (visual_request_id_.empty()) {
      return;
    }
    try {
      const auto status = Json::parse(payload);
      const auto value = status.value("status", "");
      if (status.value("request_id", "") == visual_request_id_ &&
          status.value("plan_id", "") == plan_id_ &&
          (value == "found" || value == "not_found" || value == "uncertain")) {
        visual_terminal_ = status;
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
    if (root_type_ == "Sequence") {
      while (results_.size() < steps_.size()) {
        const auto& step = steps_[results_.size()];
        auto skipped = step.original;
        skipped["status"] = "skipped";
        skipped["action"] = step.name;
        skipped["reason"] = "prior_action_not_completed";
        skipped["node_status"] = "idle";
        results_.push_back(std::move(skipped));
      }
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
    const auto index = *current_step_;
    auto result = steps_[index].original;
    result["status"] = status;
    result["action"] = steps_[index].name;
    result["request_id"] = current_request_id_;
    result["attempt"] = attempt_counts_[index];
    result["node_status"] = status == "completed" ? "success" :
        status == "interrupted" ? "halted" : "failure";
    if (!reason.empty()) {
      result["reason"] = reason;
    }
    if (!executor.empty()) {
      result["executor"] = executor;
    }
    attempt_results_[index].push_back(result);
    if (status == "completed") {
      result["attempts"] = attempt_results_[index];
      results_.push_back(std::move(result));
    }
    current_step_.reset();
    current_request_id_.clear();
    current_terminal_.reset();
  }

  void record_visual_result(bool found) {
    Json result = {
        {"name", "search_environment"},
        {"action", "search_environment"},
        {"status", "completed"},
        {"node_status", "success"},
        {"found", found},
        {"target", visual_target_},
        {"attempts", visual_attempt_},
        {"evidence", last_visual_evidence_}};
    if (found) {
      result["response"] = last_visual_response_.empty()
          ? "我找到" + visual_target_ + "了。"
          : last_visual_response_;
    } else {
      result["response"] = "我查看了" + std::to_string(visual_attempt_) +
          "个方向，仍然没有找到" + visual_target_ + "。";
      result["search_status"] = last_visual_status_;
    }
    results_.push_back(std::move(result));
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
    attempt_results_.clear();
    attempt_counts_.clear();
    plan_id_.clear();
    current_step_.reset();
    current_request_id_.clear();
    current_terminal_.reset();
    cancelled_ = false;
    stop_sent_ = false;
    root_type_.clear();
    visual_target_.clear();
    visual_question_.clear();
    visual_max_views_ = 0;
    visual_attempt_ = 0;
    visual_request_id_.clear();
    visual_terminal_.reset();
    last_visual_status_.clear();
    last_visual_evidence_.clear();
    last_visual_response_.clear();
  }

  BT::BehaviorTreeFactory factory_;
  std::map<std::string, Json> action_skills_;
  std::optional<BT::Tree> tree_;
  std::vector<PlanStep> steps_;
  Json results_ = Json::array();
  std::vector<Json> attempt_results_;
  std::vector<unsigned> attempt_counts_;
  std::string plan_id_;
  std::string root_type_;
  std::string visual_search_tree_path_;
  std::optional<unsigned> current_step_;
  std::string current_request_id_;
  std::optional<Json> current_terminal_;
  std::chrono::steady_clock::time_point step_deadline_;
  std::chrono::steady_clock::time_point visual_deadline_;
  std::chrono::duration<double> action_timeout_{20.0};
  bool cancelled_ = false;
  bool stop_sent_ = false;
  std::string visual_target_;
  std::string visual_question_;
  unsigned visual_max_views_ = 0;
  unsigned visual_attempt_ = 0;
  std::string visual_request_id_;
  std::optional<Json> visual_terminal_;
  std::string last_visual_status_;
  std::string last_visual_evidence_;
  std::string last_visual_response_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr action_command_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr tree_status_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr execute_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr cancel_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr action_status_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr visual_search_request_pub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr visual_search_status_sub_;
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

BT::NodeStatus RecoveryStopNode::tick() {
  const auto input = getInput<unsigned>("step_index");
  if (!input) {
    return BT::NodeStatus::FAILURE;
  }
  owner_->recover_step(input.value());
  return BT::NodeStatus::FAILURE;
}

BT::NodeStatus VisualSearchNode::onStart() {
  return owner_->start_visual_search() ? BT::NodeStatus::RUNNING
                                        : BT::NodeStatus::FAILURE;
}

BT::NodeStatus VisualSearchNode::onRunning() {
  return owner_->poll_visual_search();
}

void VisualSearchNode::onHalted() { owner_->halt_visual_search(); }

BT::NodeStatus SearchNotFoundNode::tick() {
  owner_->report_target_not_found();
  return BT::NodeStatus::SUCCESS;
}

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
