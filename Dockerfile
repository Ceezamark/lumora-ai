FROM python:3.10

# Install OpenCV dependencies
RUN apt-get install -y libgl1

# Copy your app code
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Command to run your FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
