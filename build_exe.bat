@echo off
REM Build DocFinderRAG.exe as a single self-contained Windows executable.
REM Run this from the project folder:  build_exe.bat

echo === Installing build dependencies ===
python -m pip install --upgrade pyinstaller pypdf python-docx openpyxl python-pptx xlrd

echo === Building executable ===
python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --windowed ^
  --name DocFinderRAG ^
  --icon icon.ico ^
  --collect-all pptx ^
  --hidden-import openpyxl ^
  --hidden-import xlrd ^
  app.py

echo.
echo === Done ===
echo The executable is at:  dist\DocFinderRAG.exe
pause
