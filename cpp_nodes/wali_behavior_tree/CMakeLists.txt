cmake_minimum_required(VERSION 3.8)
project(wali_behavior_tree)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# BehaviorTree.CPP 4.9.1 packages on ARM64 Humble install the shared library
# under lib/<multiarch>, while their generated ament config only searches lib.
# Resolve the public header/library directly so the package works with both the
# affected multiarch layout and the normal upstream layout.
set(BTCPP_LIBRARY_HINTS)
set(BTCPP_INCLUDE_HINTS)
foreach(prefix IN LISTS CMAKE_PREFIX_PATH)
  list(APPEND BTCPP_LIBRARY_HINTS
    "${prefix}/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
    "${prefix}/lib")
  list(APPEND BTCPP_INCLUDE_HINTS "${prefix}/include")
endforeach()
find_path(BTCPP_INCLUDE_DIR behaviortree_cpp/bt_factory.h
  HINTS ${BTCPP_INCLUDE_HINTS}
  REQUIRED)
find_library(BTCPP_LIBRARY behaviortree_cpp
  HINTS ${BTCPP_LIBRARY_HINTS}
  REQUIRED)

add_executable(behavior_tree_node src/behavior_tree_node.cpp)
ament_target_dependencies(behavior_tree_node rclcpp std_msgs)
target_include_directories(behavior_tree_node PRIVATE ${BTCPP_INCLUDE_DIR})
target_link_libraries(behavior_tree_node
  ${BTCPP_LIBRARY}
  nlohmann_json::nlohmann_json)
target_compile_options(behavior_tree_node PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS behavior_tree_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
