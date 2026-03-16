cmake_minimum_required(VERSION 3.20)
project(aniso LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# --- Eigen (header-only) ---
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(eigen)

# --- yaml-cpp ---
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        0.8.0
    GIT_SHALLOW    TRUE
)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

# --- GLFW ---
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# --- Dear ImGui ---
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.8
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(imgui)

add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC glfw)
if(WIN32)
    target_link_libraries(imgui_lib PUBLIC opengl32)
endif()

# --- implot ---
FetchContent_Declare(
    implot
    GIT_REPOSITORY https://github.com/epezent/implot.git
    GIT_TAG        v0.16
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(implot)

add_library(implot_lib STATIC
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
)
target_include_directories(implot_lib PUBLIC ${implot_SOURCE_DIR})
target_link_libraries(implot_lib PUBLIC imgui_lib)

# --- CLI executable ---
add_executable(aniso src/main.cpp)
target_include_directories(aniso PRIVATE include)
target_link_libraries(aniso PRIVATE Eigen3::Eigen yaml-cpp::yaml-cpp)

# --- GUI executable ---
add_executable(aniso_gui src/gui_main.cpp)
target_include_directories(aniso_gui PRIVATE include)
target_link_libraries(aniso_gui PRIVATE Eigen3::Eigen yaml-cpp::yaml-cpp implot_lib)
if(MSVC)
    set_target_properties(aniso_gui PROPERTIES
        LINK_FLAGS "/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
endif()

if(MSVC)
    target_compile_options(aniso     PRIVATE /W4 /permissive-)
    target_compile_options(aniso_gui PRIVATE /W4 /permissive-)
else()
    target_compile_options(aniso     PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(aniso_gui PRIVATE -Wall -Wextra -Wpedantic)
endif()

# --- Tests ---
enable_testing()

add_executable(test_scalar tests/test_scalar_limit.cpp)
target_include_directories(test_scalar PRIVATE include)
target_link_libraries(test_scalar PRIVATE Eigen3::Eigen yaml-cpp::yaml-cpp)

add_test(NAME scalar_limit COMMAND test_scalar)
