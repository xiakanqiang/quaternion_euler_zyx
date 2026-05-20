@echo off
cd /d "%~dp0"
python -m PyInstaller --onefile --windowed --name QuaternionEuler --clean --noconfirm quaternion_euler_zyx.py
echo.
echo 输出目录: dist\QuaternionEuler.exe
pause
