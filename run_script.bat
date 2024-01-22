@echo off
call C:\ProgramData\anaconda3\Scripts\activate.bat active_alignment
C:\ProgramData\anaconda3\envs\active_alignment\python active_alignment_gui.py
call C:\ProgramData\anaconda3\Scripts\deactivate.bat