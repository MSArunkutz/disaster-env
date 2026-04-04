@echo off
for /f "tokens=1,2 delims==" %%a in (.env) do set %%a=%%b

set API_BASE_URL=%GROQ_API_BASE_URL%
set MODEL_NAME=%GROQ_MODEL_NAME%
set HF_TOKEN=%GROQ_API_KEY%

docker build -t disaster-env .
docker run -p %PORT%:%PORT% ^
  -e API_BASE_URL=%API_BASE_URL% ^
  -e MODEL_NAME=%MODEL_NAME% ^
  -e HF_TOKEN=%HF_TOKEN% ^
  -e DIFFICULTY=%DIFFICULTY% ^
  -e PORT=%PORT% ^
  disaster-env