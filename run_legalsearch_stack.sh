#!/bin/bash

# Start the FastAPI backend (in background)
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait a moment to ensure FastAPI is up
sleep 2

# Start the Gradio frontend
python gradio_app.py

# Cleanup: stop FastAPI server on exit
kill $FASTAPI_PID
