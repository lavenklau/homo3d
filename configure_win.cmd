@echo off
setlocal enabledelayedexpansion

rem Configure homo3d on Windows with the MSVC toolchain and vcpkg.
rem Run this once, then use build_win.cmd for every rebuild.
rem
rem Environment overrides:
rem   VCVARS      full path to vcvars64.bat  (auto-detected with vswhere)
rem   VCPKG_ROOT  vcpkg installation root    (default: C:\vcpkg)

if not defined VCVARS call :detect_vcvars
if not defined VCVARS (
    echo [configure_win] Could not locate vcvars64.bat.
    echo [configure_win] Set VCVARS to its full path and retry.
    exit /b 1
)
if not exist "!VCVARS!" (
    echo [configure_win] vcvars64.bat not found at: !VCVARS!
    exit /b 1
)

if not defined VCPKG_ROOT set "VCPKG_ROOT=C:\vcpkg"
set "VCPKG_TOOLCHAIN=!VCPKG_ROOT!\scripts\buildsystems\vcpkg.cmake"
if not exist "!VCPKG_TOOLCHAIN!" (
    echo [configure_win] vcpkg toolchain not found at: !VCPKG_TOOLCHAIN!
    echo [configure_win] Set VCPKG_ROOT to your vcpkg installation and retry.
    exit /b 1
)

call "!VCVARS!" >nul
if errorlevel 1 exit /b 1

cmake -S "%~dp0." -B "%~dp0build" -G "NMake Makefiles" ^
      -DCMAKE_TOOLCHAIN_FILE="!VCPKG_TOOLCHAIN!" ^
      -DVCPKG_TARGET_TRIPLET=x64-windows
exit /b !errorlevel!

:detect_vcvars
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "!VSWHERE!" exit /b 0
for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if defined VSINSTALL set "VCVARS=!VSINSTALL!\VC\Auxiliary\Build\vcvars64.bat"
exit /b 0
