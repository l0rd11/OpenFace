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
    src/Face_utils.cpp
	src/FaceAnalyser.cpp
	src/FaceAnalyserParameters.cpp
	src/PAW.cpp
	src/PDM.cpp		
	src/SVM_dynamic_lin.cpp
	src/SVM_static_lin.cpp
	src/SVR_dynamic_lin_regressors.cpp
	src/SVR_static_lin_regressors.cpp
)

SET(HEADERS
    include/Face_utils.h	
	include/FaceAnalyser.h
	include/FaceAnalyserParameters.h
	include/PAW.h
	include/PDM.h		
	include/SVM_dynamic_lin.h
	include/SVM_static_lin.h
	include/SVR_dynamic_lin_regressors.h
	include/SVR_static_lin_regressors.h
)

include_directories(./include)
include_directories(${FACEANALYSER_SOURCE_DIR}/include)

add_library( FaceAnalyser ${SOURCE} ${HEADERS})
add_library( FaceAnalyserShared SHARED ${SOURCE} ${HEADERS} )
target_link_libraries( FaceAnalyserShared ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES} )

install (TARGETS FaceAnalyser DESTINATION lib)
install (TARGETS FaceAnalyserShared LIBRARY DESTINATION lib PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ GROUP_WRITE WORLD_WRITE WORLD_EXECUTE WORLD_READ)
install (FILES ${HEADERS} DESTINATION include/OpenFace)
