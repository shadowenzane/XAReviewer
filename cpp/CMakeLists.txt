cmake_minimum_required(VERSION 3.16)
project(XAReviewerCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# Qt（5/6 兼容，优先 6）
# ---------------------------------------------------------------------------
find_package(Qt6 QUIET COMPONENTS Core Gui Widgets)
if(Qt6_FOUND)
  set(QT_TARGETS Qt6::Core Qt6::Gui Qt6::Widgets)
  set(QT_LIBS Qt6::Core Qt6::Gui Qt6::Widgets)
  message(STATUS "Qt version: 6")
else()
  find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)
  set(QT_TARGETS Qt5::Core Qt5::Gui Qt5::Widgets)
  set(QT_LIBS Qt5::Core Qt5::Gui Qt5::Widgets)
  message(STATUS "Qt version: 5")
endif()

# ---------------------------------------------------------------------------
# OpenCV
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)

# ---------------------------------------------------------------------------
# DCMTK
# ---------------------------------------------------------------------------
find_package(DCMTK REQUIRED)

# ---------------------------------------------------------------------------
# ITK：优先直连库（Debian 的 ITKConfig 会强制加载 VTK/MPI）
# ---------------------------------------------------------------------------
find_path(ITK_INCLUDE_DIR itkImage.h
  PATH_SUFFIXES include/ITK-5.4 include/ITK-5.3 include/ITK-5.2)
find_library(ITK_COMMON_LIBRARY NAMES ITKCommon ITKCommon-5.4 ITKCommon-5.3)
find_library(ITK_SMOOTHING_LIBRARY NAMES ITKSmoothing ITKSmoothing-5.4 ITKSmoothing-5.3)
find_library(ITKVNL_LIBRARY NAMES itkvnl itkvnl-5.4 itkvnl-5.3)
find_library(ITKVNL_ALGO_LIBRARY NAMES itkvnl_algo itkvnl_algo-5.4 itkvnl_algo-5.3)
find_library(ITKV3P_NETLIB_LIBRARY NAMES itkv3p_netlib itkv3p_netlib-5.4 itkv3p_netlib-5.3)
if(ITK_INCLUDE_DIR AND ITK_COMMON_LIBRARY AND ITK_SMOOTHING_LIBRARY)
  set(ITK_LIBRARIES ${ITK_COMMON_LIBRARY} ${ITK_SMOOTHING_LIBRARY}
      ${ITKVNL_LIBRARY} ${ITKVNL_ALGO_LIBRARY} ${ITKV3P_NETLIB_LIBRARY})
  set(ITK_INCLUDE_DIRS ${ITK_INCLUDE_DIR})
  message(STATUS "ITK: direct libs ${ITK_LIBRARIES}")
else()
  find_package(ITK 5.1 REQUIRED COMPONENTS ITKCommon ITKSmoothing)
  include(${ITK_USE_FILE})
endif()

# ---------------------------------------------------------------------------
# 库目标
# ---------------------------------------------------------------------------
add_library(xareviewer_core STATIC
  src/core/DicomSeries.cpp
  src/core/DsaSequence.cpp
  src/core/ImagePipeline.cpp
  src/core/PacsFinder.cpp
)
target_link_libraries(xareviewer_core PUBLIC
  ${QT_LIBS} ${OpenCV_LIBS} ${DCMTK_LIBRARIES} ${ITK_LIBRARIES})
target_include_directories(xareviewer_core PUBLIC src ${ITK_INCLUDE_DIRS})

add_library(xareviewer_ui STATIC
  src/ui/Theme.cpp
  src/ui/MeasurementTool.cpp
  src/ui/ImageViewer.cpp
  src/ui/ControlPanel.cpp
  src/ui/PacsQueryDialog.cpp
  src/ui/MainWindow.cpp
)
target_link_libraries(xareviewer_ui PUBLIC xareviewer_core)

add_library(xareviewer_export STATIC
  src/export/Exporter.cpp
)
target_link_libraries(xareviewer_export PUBLIC xareviewer_core)

# ---------------------------------------------------------------------------
# 可执行
# ---------------------------------------------------------------------------
add_executable(xareviewer src/main.cpp)
target_link_libraries(xareviewer PRIVATE xareviewer_ui xareviewer_export)

# ---------------------------------------------------------------------------
# 测试（冒烟）
# ---------------------------------------------------------------------------
enable_testing()

add_executable(core_smoke tests/core_smoke.cpp)
target_link_libraries(core_smoke PRIVATE xareviewer_core xareviewer_export)

add_executable(ui_smoke tests/ui_smoke.cpp)
target_link_libraries(ui_smoke PRIVATE xareviewer_ui xareviewer_export)
set_target_properties(ui_smoke PROPERTIES AUTOMOC ON)

add_test(NAME core_smoke COMMAND core_smoke)
add_test(NAME ui_smoke COMMAND ui_smoke)
