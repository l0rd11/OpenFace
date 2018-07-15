include_directories(${BOOST_INCLUDE_DIR})

SET(SOURCE
    src/ImageCapture.cpp
	src/RecorderCSV.cpp
    src/RecorderHOG.cpp
	src/RecorderOpenFace.cpp
    src/RecorderOpenFaceParameters.cpp
	src/SequenceCapture.cpp
	src/VisualizationUtils.cpp
	src/Visualizer.cpp
)

SET(HEADERS
    include/ImageCapture.h	
    include/RecorderCSV.h
	include/RecorderHOG.h
    include/RecorderOpenFace.h
	include/RecorderOpenFaceParameters.h
	include/SequenceCapture.h
	include/VisualizationUtils.h
	include/Visualizer.h	
)

include_directories(./include)
include_directories(${UTILITIES_SOURCE_DIR}/include)

add_library( Utilities ${SOURCE} ${HEADERS})
add_library( UtilitiesShared SHARED ${SOURCE} ${HEADERS} )

target_link_libraries( UtilitiesShared ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES} )



install (TARGETS Utilities DESTINATION lib)
install (TARGETS UtilitiesShared LIBRARY DESTINATION lib PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ GROUP_WRITE WORLD_WRITE WORLD_EXECUTE WORLD_READ)

install (FILES ${HEADERS} DESTINATION include/OpenFace)
