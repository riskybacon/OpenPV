## See http://www.cmake.org/Wiki/CMake_Useful_Variables for more variables to set
cmake_minimum_required(VERSION 2.8.10)

## Version 0.1.0.0
set(PV_VERSION_MAJOR 0)
set(PV_VERSION_MINOR 1)
set(PV_VERSION_PATCH 0)
set(PV_VERSION_TWEAK 0)

# Nicer for in-source builds. Puts the library into a subdir
set(LIBRARY_OUTPUT_PATH ${PV_LIBRARY_DIR})

set(PV_IO_DIR "${PV_LIBRARY_DIR}/io")
set(PV_PARSER_DIR "${PV_IO_DIR}/parser-generated")
set(PV_PARAM_LEXER_C "${PV_PARSER_DIR}/param_lexer.c")
set(PV_PARAM_PARSER_CPP "${PV_PARSER_DIR}/param_parser.cpp")
set(PV_PARAM_PARSER_HPP "${PV_PARSER_DIR}/param_parser.hpp")

set(PV_LIBRARY_NAME "pv")

# Create directory structure for lexer and parser output
file(MAKE_DIRECTORY ${PV_IO_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${PV_IO_DIR})
file(MAKE_DIRECTORY ${PV_PARSER_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${PV_PARSER_DIR})

# Lex/Yacc for parser
if (NOT BISON_PARSER_DEFINED)
  find_package(BISON REQUIRED)
  BISON_TARGET(PARSER ${PV_SOURCE_DIR}/io/parser/params.y ${PV_PARAM_PARSER_CPP} DEFINES_FILE ${PV_PARAM_PARSER_HPP} COMPILE_FLAGS "-y")
endif (NOT BISON_PARSER_DEFINED)

if (NOT FLEX_SCANNER_DEFINED)
  find_package(FLEX REQUIRED)
  FLEX_TARGET(SCANNER ${PV_SOURCE_DIR}/io/parser/params.l ${PV_PARAM_LEXER_C})
  ADD_FLEX_BISON_DEPENDENCY(SCANNER PARSER)
endif (NOT FLEX_SCANNER_DEFINED)

# Library source subdirectories
set(SRC_SUBDIRS
  src/arch/cuda
  src/arch/mpi
  src/bindings
  src/columns
  src/connections
  src/cudakernels
  src/include
  src/io
  src/kernels
  src/layers
  src/normalizers
  src/utils
  src/weightinit
)

foreach(SUBDIR ${SRC_SUBDIRS})
  include("${SUBDIR}/CMakeLists.txt")
endforeach()

<<<<<<< HEAD
# Read in file and parse by line
file(READ "${PV_SRC_LIST}" contents)
STRING(REGEX REPLACE "\n" ";" contents "${contents}")

# Find PetaVision c/cpp files
foreach(LIN ${contents})
   file(GLOB libSrcCPP ${LIN}/*.cpp)
   file(GLOB libSrcC ${LIN}/*.c)
   set(PVLibSrc ${PVLibSrc} ${libSrcCPP})
   set(PVLibSrc ${PVLibSrc} ${libSrcC})

   if (PV_INCLUDE_HEADERS)
      file(GLOB libSrcHPP ${LIN}/*.h)
      file(GLOB libSrcH ${LIN}/*.hpp)
      set(PVLibSrc ${PVLibSrc} ${libSrcHPP})
      set(PVLibSrc ${PVLibSrc} ${libSrcH})
   endif()

   if(PV_USE_CUDA)
      file(GLOB_RECURSE libSrcCU ${LIN}/*.cu)
      set(PVLibSrcCu ${PVLibSrcCu} ${libSrcCU})
   endif(PV_USE_CUDA)
endforeach(LIN ${contents})

find_package(Git)
if (GIT_FOUND)
   unset(PV_GIT_REVISION)
   # Get current commit
   execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                   RESULT_VARIABLE PV_CURRENT_COMMIT_RESULT
                   OUTPUT_VARIABLE PV_CURRENT_COMMIT
                   ERROR_VARIABLE PV_CURRENT_COMMIT_ERROR
                   OUTPUT_STRIP_TRAILING_WHITESPACE)
   if (${PV_CURRENT_COMMIT_RESULT} EQUAL 0)
      # Get commit hash and date commit was authored
      execute_process(COMMAND "${GIT_EXECUTABLE}" log -n 1 "--format=%H (%ad)" "${PV_CURRENT_COMMIT}"
                      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      RESULT_VARIABLE PV_GIT_REVISION_RESULT
                      OUTPUT_VARIABLE PV_GIT_REVISION
                      ERROR_VARIABLE PV_GIT_REVISION_ERROR
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      set(PV_GIT_REVISION "git repository version ${PV_GIT_REVISION}")
      # See if there are any local changes
      execute_process(COMMAND "${GIT_EXECUTABLE}" status -s
                      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      RESULT_VARIABLE PV_GIT_STATUS_RESULT
                      OUTPUT_VARIABLE PV_GIT_STATUS_OUTPUT
                      ERROR_VARIABLE VP_GIT_STATUS_ERROR
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      string(LENGTH "${PV_GIT_STATUS_OUTPUT}" PV_STATUS_LENGTH)
      if(NOT ("${PV_STATUS_LENGTH}" EQUAL 0))
         set(PV_GIT_REVISION "${PV_GIT_REVISION} with local modifications")
      endif(NOT ("${PV_STATUS_LENGTH}" EQUAL 0))
   else (${PV_CURRENT_COMMIT_RESULT} EQUAL 0)
      unset(PV_GIT_REVISION)
   endif (${PV_CURRENT_COMMIT_RESULT} EQUAL 0)
endif (GIT_FOUND)

find_package(Subversion)
if (Subversion_FOUND)
   unset(PV_SVN_REVISION)
   set(SVNVERSION_COMMAND "${Subversion_SVN_EXECUTABLE}version")
   # !! lack of space between ${Subversion_SVN_EXECUTABLE} and version is deliberate:
   # !! Subversion_SVN_EXECUTABLE returns svn, and we want to call svnversion.
   execute_process(COMMAND "${SVNVERSION_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}"
                   OUTPUT_VARIABLE PV_SVN_REVISION
                   ERROR_VARIABLE PV_SVN_ERROR
                   RESULT_VARIABLE PV_SVN_RESULT
                   OUTPUT_STRIP_TRAILING_WHITESPACE)
   if (${PV_SVN_RESULT} EQUAL 0)
      if ("${PV_SVN_REVISION}" STREQUAL "Unversioned directory")
         unset(PV_SVN_REVISION)
      else ("${PV_SVN_REVISION}" STREQUAL "Unversioned directory")
         string(REGEX REPLACE "M$" " with local modifications" PV_SVN_REVISION "${PV_SVN_REVISION}")
         set(PV_SVN_REVISION "svn repository version ${PV_SVN_REVISION}")
      endif ("${PV_SVN_REVISION}" STREQUAL "Unversioned directory")
   else (${PV_SVN_RESULT} EQUAL 0)
      unset(PV_SVN_REVISION)
   endif (${PV_SVN_RESULT} EQUAL 0)
else (Subversion_FOUND)
   unset(PV_SVN_REVISION)
endif (Subversion_FOUND)

if (DEFINED PV_GIT_REVISION)
   set(PV_REVISION "${PV_GIT_REVISION}")
endif (DEFINED PV_GIT_REVISION)
if (DEFINED PV_SVN_REVISION)
   set(PV_REVISION "${PV_SVN_REVISION}")
endif (DEFINED PV_SVN_REVISION)
if (DEFINED PV_SVN_REVISION AND DEFINED PV_GIT_REVISION) # This probably shouldn't happen
   set(PV_REVISION "${PV_GIT_REVISION}/${PV_SVN_REVISION}")
endif (DEFINED PV_SVN_REVISION AND DEFINED PV_GIT_REVISION)
if (NOT (DEFINED PV_SVN_REVISION OR DEFINED PV_GIT_REVISION))
   set(PV_REVISION "unknown version")
endif (NOT (DEFINED PV_SVN_REVISION OR DEFINED PV_GIT_REVISION))
=======
set(PVLibSrc ${PVLibSrcCpp} ${PVLibSrcHpp} ${PVLibSrcC} ${PVLibSrcH})

include(PVGitRevision)
>>>>>>> master

# Header file to pass CMake settings to source code
if(PV_USE_OPENMP)
  # C++ code needs PV_USE_OPENMP_THREADS defined
  set(PV_USE_OPENMP_THREADS ON)
endif()

file(MAKE_DIRECTORY ${PV_CONFIG_FILE_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${PV_CONFIG_FILE_DIR})
configure_file (
   "cMakeHeader.template"
   "${PV_CONFIG_FILE_DIR}/cMakeHeader.h"
)

if (PV_BUILD_SHARED) 
  message(STATUS "Building shared library")
  if (APPLE)
    # Like LD_LIBRARY_PATH, but for OS X
    set(CMAKE_MACOSX_RPATH ${PV_LIBRARY_DIR})
  endif()
  set(SHARED_FLAG "SHARED")
else()
  message(STATUS "Building static library")
  set(SHARED_FLAG "STATIC")
endif()

include_directories(${PV_CONFIG_FILE_DIR})
include_directories(${PV_INCLUDE_DIR})

if (PV_USE_GDAL AND GDAL_FOUND)
  include_directories(${GDAL_INCLUDE_DIR})
endif()

if (PV_USE_CUDA AND CUDNN_FOUND)
  include_directories(${CUDA_TOOLKIT_INCLUDE})
  include_directories(${CUDNN_INCLUDE_DIR})
endif()

if (PV_USE_MPI AND MPI_FOUND)
  include_directories(${MPI_CXX_INCLUDE_PATH})
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinRelSize")
  list(APPEND CMAKE_CXX_FLAGS ${PV_COMPILE_FLAGS_RELEASE})
else()
  list(APPEND CMAKE_CXX_FLAGS ${PV_COMPILE_FLAGS_DEBUG})
endif()

if(PV_USE_CUDA AND CUDA_FOUND AND CUDNN_FOUND)
  list(APPEND CMAKE_CXX_FLAGS ${PV_NVCC_FLAGS})
  STRING(REGEX REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  cuda_add_library(${PV_LIBRARY_NAME} ${SHARED_FLAG} ${PVLibSrc} ${BISON_PARSER_OUTPUTS} ${FLEX_SCANNER_OUTPUTS} ${PVLibSrcCu})
else()
  # Add  PetaVision library target, no CUDA
  STRING(REGEX REPLACE ";" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  add_library(${PV_LIBRARY_NAME} ${SHARED_FLAG} ${PVLibSrc} ${BISON_PARSER_OUTPUTS} ${FLEX_SCANNER_OUTPUTS})
endif()

