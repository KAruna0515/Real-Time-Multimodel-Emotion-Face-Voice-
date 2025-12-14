@echo off
cd /d %~dp0
call ..\.venv\Scripts\activate.bat
python demo/live_camera_client.py
pause
