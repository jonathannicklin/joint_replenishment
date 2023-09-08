
#for succesfully linking DLL on windows.
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
#to enable linking static libs into shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

#for succesfully linking on linux to desired paths, i.e. to enable GPU
#note that this will require some manual steps and is not enabled for now. 
#set(CMAKE_SKIP_RPATH TRUE)
if(MSVC)
set(CMAKE_CXX_FLAGS "$CMAKE_CXX_FLAGS /EHsc")
endif()

set(stageDir ${CMAKE_CURRENT_BINARY_DIR})
include(GNUInstallDirs)
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${stageDir}/${CMAKE_INSTALL_BINDIR})
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${stageDir}/${CMAKE_INSTALL_LIBDIR})
endif()
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${stageDir}/${CMAKE_INSTALL_LIBDIR})
endif()
