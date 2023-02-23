# Use the official Python image as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenCV
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python-headless

# Set environment variable to run main.py
ENV PYTHONPATH=/app
CMD ["python", "app_detection.py" ]
