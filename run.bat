@echo off
cd /d "E:\1_RecSys\Recommender"
call venv\Scripts\activate
uvicorn app:app --reload
pause
