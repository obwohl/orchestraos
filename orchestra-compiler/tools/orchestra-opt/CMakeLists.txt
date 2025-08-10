add_executable(orchestra-opt main.cpp)

target_link_libraries(orchestra-opt
  PRIVATE
  Orchestra
  MLIRFuncDialect
  MLIRMlirOptMain
)
