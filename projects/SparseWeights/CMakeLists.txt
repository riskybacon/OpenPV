include(PVAddExecutable)
# Name the target based on the directory name
get_filename_component(TARGET ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(RUN_TARGET ${TARGET}_run)

pv_add_executable(${TARGET} SRC src/main.cpp)
add_dependencies(${TARGET} pv)

# If out-of-source build, create a symlink to the input dir
if (NOT ${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
  set(LINK_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/input")
  set(LINK_DEST "${CMAKE_CURRENT_BINARY_DIR}/input")
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${LINK_SOURCE} ${LINK_DEST})
endif()
 
#add_custom_target(${RUN_TARGET} COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_CURRENT_BINARY_DIR} ${TARGET} -p input/LCATest.params -t -l log)



