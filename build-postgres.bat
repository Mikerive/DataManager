@echo off
setlocal

:: Load MSVC environment (Adjust path if needed)
call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

:: Move to PostgreSQL source directory
cd /d "%~dp0"

:: Configure PostgreSQL
configure --prefix="C:\pgsql" --enable-debug

:: Compile using nmake
nmake /f Makefile

:: Show success message
echo PostgreSQL Build Completed Successfully!
