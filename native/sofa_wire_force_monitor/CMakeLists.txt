cmake_minimum_required(VERSION 3.16)
project(SofaWireForceMonitor LANGUAGES CXX)

if(NOT DEFINED SOFA_ROOT)
  if(DEFINED ENV{SOFA_ROOT})
    set(SOFA_ROOT "$ENV{SOFA_ROOT}")
  endif()
endif()

if(NOT DEFINED SOFA_ROOT)
  message(FATAL_ERROR "SOFA_ROOT is not set. Pass -DSOFA_ROOT=/path/to/SOFA_v23.06.00_Linux.")
endif()

set(_SOFA_CONFIG_FILE "${SOFA_ROOT}/lib/cmake/Sofa.Config/Sofa.ConfigConfig.cmake")
if(NOT EXISTS "${_SOFA_CONFIG_FILE}")
  message(FATAL_ERROR "Sofa.ConfigConfig.cmake not found at ${_SOFA_CONFIG_FILE}")
endif()

file(GLOB _SOFA_CMAKE_DIRS "${SOFA_ROOT}/lib/cmake/*")
list(PREPEND CMAKE_PREFIX_PATH ${_SOFA_CMAKE_DIRS})

include("${_SOFA_CONFIG_FILE}")
sofa_find_package(Sofa.Core REQUIRED)
sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.Component.StateContainer REQUIRED)
sofa_find_package(Sofa.Component.Mapping.Linear REQUIRED)

add_library(SofaWireForceMonitor SHARED
  src/WireWallContactExport.cpp
  src/WireWallForceMonitor.cpp
  src/initSofaWireForceMonitor.cpp
)

add_library(SofaWireForceMonitor::SofaWireForceMonitor ALIAS SofaWireForceMonitor)

target_compile_features(SofaWireForceMonitor PRIVATE cxx_std_17)
target_compile_definitions(SofaWireForceMonitor PRIVATE SOFA_BUILD_SOFA_WIRE_FORCE_MONITOR)

target_include_directories(SofaWireForceMonitor
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(SofaWireForceMonitor
  PUBLIC
    Sofa.Core
    Sofa.Simulation.Core
    Sofa.Component.Mapping.Linear
    Sofa.Component.StateContainer
)

# The runtime process (steve-compare) often runs from a conda env with an older
# libstdc++. Build the plugin with static C++ runtime to avoid GLIBCXX version
# mismatches at SOFA plugin load time.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_options(SofaWireForceMonitor PRIVATE -static-libstdc++ -static-libgcc)
endif()

set_target_properties(SofaWireForceMonitor PROPERTIES
  OUTPUT_NAME "SofaWireForceMonitor"
  POSITION_INDEPENDENT_CODE ON
)

if(DEFINED SOFA_ROOT)
  set_target_properties(SofaWireForceMonitor PROPERTIES
    BUILD_RPATH "${SOFA_ROOT}/lib;${SOFA_ROOT}/plugins/BeamAdapter/lib"
  )
endif()
