@ECHO OFF

REM Command file for Sphinx documentation
REM Adapted from the default 'sphinx-quickstart'

set SPHINXBUILD=sphinx-build
set BUILDDIR=build
set SOURCEDIR=source
set SPHINXOPTS=

if "%1"=="html" (
    %SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
) else if "%1"=="clean" (
    rmdir /S /Q %BUILDDIR%
) else (
    echo "Please use 'make html' or 'make clean'"
)
