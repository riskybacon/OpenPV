include(PVAddExecutable)

get_filename_component(PV_PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")

set(${PV_PROJECT_NAME}_SRCCPP
  src/ConvertFromTable.cpp
  src/HarnessCustomGroupHandler.cpp
  src/LocalizationProbe.cpp
  src/MaskFromMemoryBuffer.cpp
  src/main.cpp
)

pv_add_executable(${PV_PROJECT_NAME}
  SRC ${${PV_PROJECT_NAME}_SRCCPP}
  OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}"
)

add_dependencies(${PV_PROJECT_NAME} pv)
