echo 'Starting server using uvicorn'
uvicorn app.api:app  --host 0.0.0.0 --port 8000  --reload  --reload-dir . --reload-dir app 