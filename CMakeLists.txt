cmake_minimum_required(VERSION 3.22)
project(SMDP_CUDA CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
add_executable(SMDP_CUDA smdp.cu visualize.cpp main.cpp)

if(UNIX)
    # https://github.com/microsoft/vcpkg/issues/10949
    find_package(OpenGL REQUIRED)
    target_link_libraries(SMDP_CUDA PRIVATE OpenGL::GL)
endif()

file(COPY images DESTINATION ${CMAKE_BINARY_DIR})

find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(SMDP_CUDA PRIVATE nlohmann_json::nlohmann_json)
find_package(fmt CONFIG REQUIRED)
target_link_libraries(SMDP_CUDA PRIVATE fmt::fmt)
find_package(SFML COMPONENTS system window graphics CONFIG REQUIRED)
target_link_libraries(SMDP_CUDA PRIVATE sfml-system sfml-graphics sfml-window)
find_package(ftxui CONFIG REQUIRED)
target_link_libraries(SMDP_CUDA PRIVATE ftxui::dom ftxui::screen ftxui::component)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"/utf-8\"")

set_target_properties(SMDP_CUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
