#TBB library
include_directories(${TBB_ROOT_DIR}/include)

include_directories(${BOOST_INCLUDE_DIR})

SET(SOURCE
    src/Face_utils.cpp
	src/FaceAnalyser.cpp
	src/SVM_dynamic_lin.cpp
	src/SVM_static_lin.cpp
	src/SVR_dynamic_lin_regressors.cpp
	src/SVR_static_lin_regressors.cpp
	src/GazeEstimation.cpp
)

SET(HEADERS
    include/Face_utils.h	
	include/FaceAnalyser.h
	include/SVM_dynamic_lin.h
	include/SVM_static_lin.h
	include/SVR_dynamic_lin_regressors.h
	include/SVR_static_lin_regressors.h
	include/GazeEstimation.h
)

include_directories(./include)
include_directories(${FACEANALYSER_SOURCE_DIR}/include)
include_directories(../LandmarkDetector/include)

add_library( FaceAnalyser ${SOURCE} ${HEADERS})
add_library( FaceAnalyserShared SHARED ${SOURCE} ${HEADERS} )
target_link_libraries( FaceAnalyserShared ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES} )

install (TARGETS FaceAnalyser DESTINATION lib)
install (TARGETS FaceAnalyserShared LIBRARY DESTINATION lib PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ GROUP_WRITE WORLD_WRITE WORLD_EXECUTE WORLD_READ)
install (FILES ${HEADERS} DESTINATION include/OpenFace)
