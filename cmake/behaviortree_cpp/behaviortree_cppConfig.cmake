# Compatibility package config for ROS distributions that install the
# BehaviorTree.CPP shared library in a GNU multiarch directory while exporting
# a CMake config that only searches <prefix>/lib.

if(TARGET behaviortree_cpp::behaviortree_cpp)
  set(behaviortree_cpp_FOUND TRUE)
  return()
endif()

file(TO_CMAKE_PATH "$ENV{AMENT_PREFIX_PATH}" _btcpp_prefixes)
set(_btcpp_include_hints)
set(_btcpp_library_hints)
foreach(_prefix IN LISTS _btcpp_prefixes CMAKE_PREFIX_PATH)
  list(APPEND _btcpp_include_hints "${_prefix}/include")
  list(APPEND _btcpp_library_hints
    "${_prefix}/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
    "${_prefix}/lib/aarch64-linux-gnu"
    "${_prefix}/lib/x86_64-linux-gnu"
    "${_prefix}/lib")
endforeach()

find_path(behaviortree_cpp_INCLUDE_DIR
  NAMES behaviortree_cpp/bt_factory.h
  HINTS ${_btcpp_include_hints})
find_library(behaviortree_cpp_LIBRARY
  NAMES behaviortree_cpp
  HINTS ${_btcpp_library_hints})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(behaviortree_cpp
  REQUIRED_VARS behaviortree_cpp_INCLUDE_DIR behaviortree_cpp_LIBRARY)

if(behaviortree_cpp_FOUND)
  add_library(behaviortree_cpp::behaviortree_cpp UNKNOWN IMPORTED)
  set_target_properties(behaviortree_cpp::behaviortree_cpp PROPERTIES
    IMPORTED_LOCATION "${behaviortree_cpp_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${behaviortree_cpp_INCLUDE_DIR}")
  set(behaviortree_cpp_LIBRARIES behaviortree_cpp::behaviortree_cpp)
  set(behaviortree_cpp_INCLUDE_DIRS "${behaviortree_cpp_INCLUDE_DIR}")
endif()

unset(_btcpp_prefixes)
unset(_btcpp_include_hints)
unset(_btcpp_library_hints)
