FROM python:3.11-slim

# Set up main application directory
WORKDIR /app

# Install requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a proper directory structure where 'backend' is a package
# We'll copy everything to /app/backend to match the import patterns
RUN mkdir -p /app/backend

# Copy all backend code to /app/backend 
COPY . /app/backend/

# Create a simple main.py file in /app that imports from backend
RUN echo 'import sys; print("Starting application..."); from backend.main import app' > /app/main.py

# Create a debug script to verify imports
RUN echo "import sys; print(f'Python path: {sys.path}'); import os; print(f'Contents of /app: {os.listdir(\"/app\")}'); print(f'Current directory: {os.getcwd()}');" > /app/debug.py

# Set Python path to include the parent directory
ENV PYTHONPATH=/app

EXPOSE 8000

# Run the server with the correct import path
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]