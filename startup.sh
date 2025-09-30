
#!/bin/bash

# Install ffmpeg if not present (Render's Ubuntu environment)
if ! command -v ffmpeg &> /dev/null
then
    echo "Installing ffmpeg..."
    apt-get update && apt-get install -y ffmpeg
fi

# Start the FastAPI application with uvicorn
echo "Starting SalesBuddy Backend..."
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
