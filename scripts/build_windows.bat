@echo off
REM Build muninn.dll with MSVC
REM Requires: Visual Studio or Build Tools with MSVC
REM Usage: Run from "Developer Command Prompt for VS" or after ilammy/msvc-dev-cmd in CI

cl.exe /O2 /MT /W4 /LD /Isrc ^
    src\muninn.c ^
    src\hnsw_vtab.c ^
    src\hnsw_algo.c ^
    src\graph_tvf.c ^
    src\graph_load.c ^
    src\graph_centrality.c ^
    src\graph_community.c ^
    src\node2vec.c ^
    src\vec_math.c ^
    src\priority_queue.c ^
    src\id_validate.c ^
    /Fe:muninn.dll

if %ERRORLEVEL% neq 0 (
    echo Build failed with error %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo Built muninn.dll successfully
