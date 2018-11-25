cmake_minimum_required (VERSION 2.6)
project(dqn)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native")

# Edit these paths to point to your caffe & ALE installations
# when building caffe be sure and do "make distribute" (only necessary for old caffe)
set(ALE_ROOT_DIR "~/work/Arcade-Learning-Environment")
set(CAFFE_ROOT_DIR "~/caffe")
list(APPEND CMAKE_PREFIX_PATH ${ALE_ROOT_DIR} ${CAFFE_ROOT_DIR})

add_executable(dqn src/dqn_main.cpp)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

option(CPU_ONLY "Use CPU only for Caffe" ON)
set(CPU_ONLY ON)
if(CPU_ONLY)
  add_definitions(-DCPU_ONLY)
  message("Compile with CPU_ONLY option")
else()
  include_directories(/usr/local/cuda/include)
  message("Compile with GPU and CUDA")
endif()

option(USE_CUDNN "Use cuDNN for Caffe" OFF)
if(USE_CUDNN)
  target_link_libraries(dqn cudnn)
  message("Compile with cuDNN for Caffe")
endif()

option(USE_SDL "Use SDL for ALE" ON)
if(USE_SDL)
  find_package(SDL REQUIRED)
  add_definitions(-D__USE_SDL)
  include_directories(${SDL_INCLUDE_DIR})
  target_link_libraries(dqn ${SDL_LIBRARY} ${SDL_MAIN_LIBRARY})
endif()

find_package(CAFFE REQUIRED)
include_directories(${CAFFE_INCLUDE_DIRS})
target_link_libraries(dqn ${CAFFE_LIBRARIES})

find_package(ALE REQUIRED)
include_directories(${ALE_INCLUDE_DIRS})
target_link_libraries(dqn ${ALE_LIBRARIES})

find_package(Boost 1.40 COMPONENTS filesystem system iostreams regex REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(dqn ${Boost_LIBRARIES})

find_package(GFLAGS REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIR})
target_link_libraries(dqn ${GFLAGS_LIBRARY})

find_package(GLOG REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})
target_link_libraries(dqn ${GLOG_LIBRARIES})

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})
target_link_libraries(dqn ${PROTOBUF_LIBRARIES})
