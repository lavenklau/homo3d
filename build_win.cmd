@echo off
setlocal enabledelayedexpansion

rem Build homo3d on Windows. Run configure_win.cmd once beforehand.
rem
rem Environment overrides:
rem   VCVARS     full path to vcvars64.bat  (auto-detected with vswhere)
rem   BUILD_DIR  build directory            (default: <repo>\build)
rem   JOBS       parallel compile jobs      (default: 4, matching `make -j4`
rem              in the README; nvcc is memory hungry, so raise with care)

if not defined VCVARS call :detect_vcvars
if not defined VCVARS (
    echo [build_win] Could not locate vcvars64.bat.
    echo [build_win] Set VCVARS to its full path and retry.
    exit /b 1
)
if not exist "!VCVARS!" (
    echo [build_win] vcvars64.bat not found at: !VCVARS!
    exit /b 1
)

if not defined BUILD_DIR set "BUILD_DIR=%~dp0build"
if not exist "!BUILD_DIR!\CMakeCache.txt" (
    echo [build_win] !BUILD_DIR! is not configured yet.
    echo [build_win] Run configure_win.cmd first.
    exit /b 1
)

if not defined JOBS set "JOBS=4"

call "!VCVARS!" >nul
if errorlevel 1 exit /b 1

cmake --build "!BUILD_DIR!" --config Release --parallel !JOBS!
exit /b !errorlevel!

:detect_vcvars
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "!VSWHERE!" exit /b 0
for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if defined VSINSTALL set "VCVARS=!VSINSTALL!\VC\Auxiliary\Build\vcvars64.bat"
exit /b 0
