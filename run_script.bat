@echo off
call C:\ProgramData\anaconda3\Scripts\activate.bat active_alignment
C:\ProgramData\anaconda3\envs\active_alignment\python.exe active_alignment_gui.py
pause
call C:\ProgramData\anaconda3\Scripts\deactivate.bat