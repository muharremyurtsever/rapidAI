cmake_minimum_required(VERSION 3.27)
project(rapidai_bank LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Python + nanobind (from the project venv)
find_package(
  Python 3.12
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# MLX (headers + libmlx.dylib shipped in the pip wheel)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import mlx; print(mlx.__path__[0], end='')"
  OUTPUT_VARIABLE MLX_PY_ROOT)
find_package(MLX CONFIG REQUIRED PATHS "${MLX_PY_ROOT}/share/cmake/MLX")

nanobind_add_module(
  _rapidai_bank
  NB_STATIC
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/rapidai_bank.cpp)
target_include_directories(_rapidai_bank
                           PRIVATE "${MLX_PY_ROOT}/include/metal_cpp")
target_link_libraries(_rapidai_bank PRIVATE mlx "-framework Metal"
                      "-framework Foundation" "-framework QuartzCore")

# Make @rpath/libmlx.dylib resolvable from inside the mlx wheel
set_target_properties(
  _rapidai_bank PROPERTIES INSTALL_RPATH "${MLX_PY_ROOT}/lib"
                           BUILD_WITH_INSTALL_RPATH ON)

install(TARGETS _rapidai_bank
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
