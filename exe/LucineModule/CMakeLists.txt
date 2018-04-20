#TBB library


include_directories(${TBB_ROOT_DIR}/include)

add_library(LucineModule SHARED LucineModule.cpp)

# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)

include_directories(../../lib/local/LandmarkDetector/include)
include_directories(/usr/include/python3.6m)
include_directories(../../lib/local/FaceAnalyser/include)
include_directories(../../lib/local/GazeAnalyser/include)
include_directories(../../lib/local/Utilities/include)
target_link_libraries(LucineModule LandmarkDetector)
target_link_libraries(LucineModule FaceAnalyser)
target_link_libraries(LucineModule GazeAnalyser)
target_link_libraries(LucineModule Utilities)
target_link_libraries(LucineModule dlib)

# set_target_properties(LucineModule
                      # PROPERTIES
                      # SUFFIX .so)
set_target_properties(LucineModule
											PROPERTIES
											LIBRARY_OUTPUT_DIRECTORY ../../bin)

set_target_properties(LucineModule
											PROPERTIES
                      POSITION_INDEPENDENT_CODE ON)

set_target_properties(LucineModule PROPERTIES PREFIX "")
set_target_properties(LucineModule PROPERTIES OUTPUT_NAME "OpenFace")
target_link_libraries(LucineModule ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES} ${BLAS_LIBRARIES} boost_python3 python3.6m)

install (TARGETS LucineModule DESTINATION bin)

