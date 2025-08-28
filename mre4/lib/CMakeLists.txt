add_library(MyPass MyPass.cpp)
target_link_libraries(MyPass PUBLIC MLIRPass)
target_include_directories(MyPass PUBLIC ${CMAKE_SOURCE_DIR}/include)
