cmake_minimum_required(VERSION 3.20)
project(hy3_sidecar_io LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(hy3_sidecar_io
  hy3_sidecar_io.cpp
  hy3_expert_bank.cpp
  hy3_q4_affine.cpp
  hy3_routed_mlp.cpp
)
target_compile_options(hy3_sidecar_io PRIVATE -Wall -Wextra -Wpedantic -O3)

add_executable(hy3_route_mlp_daemon
  hy3_route_mlp_daemon.cpp
  hy3_expert_bank.cpp
  hy3_q4_affine.cpp
)
target_compile_options(hy3_route_mlp_daemon PRIVATE -Wall -Wextra -Wpedantic -O3)

add_executable(hy3_q4_affine_test
  hy3_q4_affine_test.cpp
  hy3_q4_affine.cpp
)
target_compile_options(hy3_q4_affine_test PRIVATE -Wall -Wextra -Wpedantic -O2)
add_test(NAME hy3_q4_affine_test COMMAND hy3_q4_affine_test)

if(APPLE)
  target_link_libraries(hy3_sidecar_io PRIVATE "-framework Accelerate")
  target_link_libraries(hy3_route_mlp_daemon PRIVATE "-framework Accelerate")
  target_link_libraries(hy3_q4_affine_test PRIVATE "-framework Accelerate")
endif()
