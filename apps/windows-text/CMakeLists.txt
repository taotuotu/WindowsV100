add_executable(ninfer-windows-text
  main.cpp
  no_media_decode.cpp)
target_compile_features(ninfer-windows-text PRIVATE cxx_std_20)
target_include_directories(ninfer-windows-text PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ninfer-windows-text PRIVATE ninfer_engine ninfer_text)

if(MSVC)
  target_compile_options(ninfer-windows-text PRIVATE /utf-8)
  target_compile_definitions(ninfer-windows-text PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

install(TARGETS ninfer-windows-text RUNTIME DESTINATION bin)
