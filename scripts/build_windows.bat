@echo off
REM Build muninn.dll with MSVC + llama.cpp
REM Requires: Visual Studio or Build Tools with MSVC, CMake
REM Usage: Run from "Developer Command Prompt for VS" or after ilammy/msvc-dev-cmd in CI

REM Step 1: Build llama.cpp static libraries via CMake
echo Building llama.cpp...
cmake -B vendor\llama.cpp\build -S vendor\llama.cpp ^
    -DBUILD_SHARED_LIBS=OFF ^
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON ^
    -DGGML_NATIVE=OFF ^
    -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF ^
    -DGGML_HIP=OFF -DGGML_SYCL=OFF -DGGML_OPENMP=OFF ^
    -DGGML_BACKEND_DL=OFF ^
    -DLLAMA_BUILD_COMMON=OFF ^
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF ^
    -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF ^
    -DCMAKE_BUILD_TYPE=MinSizeRel

if %ERRORLEVEL% neq 0 (
    echo CMake configure failed
    exit /b %ERRORLEVEL%
)

cmake --build vendor\llama.cpp\build --config MinSizeRel -j %NUMBER_OF_PROCESSORS%

if %ERRORLEVEL% neq 0 (
    echo llama.cpp build failed
    exit /b %ERRORLEVEL%
)

echo llama.cpp built successfully

REM Step 2: Build muninn.dll linking against llama.cpp
if not exist build mkdir build

cl.exe /O2 /MT /W4 /LD /Isrc ^
    /Ivendor\llama.cpp\include /Ivendor\llama.cpp\ggml\include ^
    src\muninn.c ^
    src\hnsw_vtab.c ^
    src\hnsw_algo.c ^
    src\graph_tvf.c ^
    src\graph_load.c ^
    src\graph_centrality.c ^
    src\graph_community.c ^
    src\graph_adjacency.c ^
    src\graph_csr.c ^
    src\graph_selector_parse.c ^
    src\graph_selector_eval.c ^
    src\graph_select_tvf.c ^
    src\node2vec.c ^
    src\vec_math.c ^
    src\priority_queue.c ^
    src\id_validate.c ^
    src\embed_gguf.c ^
    vendor\llama.cpp\build\src\MinSizeRel\llama.lib ^
    vendor\llama.cpp\build\ggml\src\MinSizeRel\ggml.lib ^
    vendor\llama.cpp\build\ggml\src\MinSizeRel\ggml-base.lib ^
    vendor\llama.cpp\build\ggml\src\MinSizeRel\ggml-cpu.lib ^
    /Fe:build\muninn.dll

if %ERRORLEVEL% neq 0 (
    echo Build failed with error %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo Built build\muninn.dll successfully
