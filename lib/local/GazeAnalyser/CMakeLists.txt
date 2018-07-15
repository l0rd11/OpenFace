#TBB library
include_directories(${TBB_ROOT_DIR}/include)
include_directories(${BLAS_ROOT_DIR})

include_directories(${BOOST_INCLUDE_DIR})

#OpenBlas library
include_directories(../../3rdParty/OpenBLAS/include)

#LandmarkDetector library
include_directories(../../local/LandmarkDetector/include)

#Utilities library
include_directories(../../local/Utilities/include)

SET(SOURCE
    src/GazeEstimation.cpp
)

SET(HEADERS
    include/GazeEstimation.h
)

include_directories(./include)
include_directories(${GAZEANALYSER_SOURCE_DIR}/include)

add_library( GazeAnalyser ${SOURCE} ${HEADERS})
add_library( GazeAnalyserShared SHARED ${SOURCE} ${HEADERS} )

target_link_libraries( GazeAnalyserShared ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES} )



install (TARGETS GazeAnalyser DESTINATION lib)
install (TARGETS GazeAnalyserShared LIBRARY DESTINATION lib PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ GROUP_WRITE WORLD_WRITE WORLD_EXECUTE WORLD_READ)

install (FILES ${HEADERS} DESTINATION include/OpenFace)
