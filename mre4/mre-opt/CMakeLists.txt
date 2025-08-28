add_executable(mre-opt main.cpp)
target_link_libraries(mre-opt PRIVATE MyPass MLIRMlirOptMain)
