cmake_minimum_required(VERSION 3.8)
project(waypoint_navigation)

# Use strict compiler warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate interfaces (actions and services)
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/NavToWaypoint.action"
  "srv/GetWaypoints.srv"
  DEPENDENCIES geometry_msgs action_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
