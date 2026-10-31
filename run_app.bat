@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo  ==========================================
echo          EstateIQ - Real Estate AI
echo  ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating a project environment...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.11 -m venv .venv >nul 2>&1
        if errorlevel 1 python -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 goto setup_error
)

call ".venv\Scripts\activate.bat"

python -c "import streamlit, tensorflow, pandas, sklearn, plotly" >nul 2>&1
if errorlevel 1 (
    echo [2/3] Installing project requirements. This may take a while on first run...
    python -m pip install --upgrade pip
    if errorlevel 1 goto setup_error
    python -m pip install -r requirements.txt
    if errorlevel 1 goto setup_error
) else (
    echo [2/3] Python dependencies are already installed.
)

echo [3/3] Starting EstateIQ in your browser...
python -m streamlit run app.py
if errorlevel 1 goto run_error
goto end

:setup_error
echo.
echo Setup could not finish. Check your Python installation and internet connection,
echo then review the error above. Python 3.11 64-bit is recommended.
pause
exit /b 1

:run_error
echo.
echo EstateIQ stopped with an error. Review the message above.
pause
exit /b 1

:end
endlocal
