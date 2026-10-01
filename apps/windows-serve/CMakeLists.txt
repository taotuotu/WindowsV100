add_executable(ninfer-windows-serve
  main.cpp
  wic_image_decode.cpp)
target_compile_features(ninfer-windows-serve PRIVATE cxx_std_20)
target_include_directories(ninfer-windows-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-windows-serve PRIVATE
  ninfer_serve
  ninfer_product_logging
  CUDA::cuda_driver
  ole32
  windowscodecs)
target_compile_definitions(ninfer-windows-serve PRIVATE
  NOMINMAX
  WIN32_LEAN_AND_MEAN
  _WIN32_WINNT=0x0A00)

install(TARGETS ninfer-windows-serve RUNTIME DESTINATION bin)
