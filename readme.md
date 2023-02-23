# Object Detection With Python

Object Detection using tensorflow, yolov3 library and opencv

This is a simple Python application that uses to detect objects in real-time using your computer's camera.


# Installation
To run this app, you'll need to have Python 3.9 and OpenCV installed. You can install OpenCV by running the following command:

**pip install opencv-python-headless**


**You'll also need to install the numpy package:**

pip install numpy==1.21.5


#Usage
To run the app, simply execute the following command:

 **python app_detection.py**

 This will start the app and display the camera feed with object detection overlays. Press 'q' to exit the app.


# Docker
You can also run this app inside a Docker container. To do so, you'll first need to build the Docker image:


**docker build -t app_detection.py .**


And then run the container:
for bash run
**docker run --rm -it app_detection**



A Dockerfile is a script that contains a set of instructions for building a Docker image. The purpose of creating a Dockerfile for an application is to create a containerized version of the application that can be run on any system that has Docker installed, without needing to worry about dependencies or environment issues.

The Dockerfile contains a series of commands that define the environment and dependencies required to run the application. These commands include instructions for installing any necessary packages, setting environment variables, and copying files into the container. Once the Dockerfile is built, it creates a Docker image, which can then be used to start a container that runs the application. This makes it easier to deploy and manage applications across multiple environments, and ensures that the application is running in a consistent and reproducible environment.



# Credit


This is a simple Python application that uses the Tensorflow YOLOv3 algorithm for object detection in real-time video streams. The app loads the YOLOv3 configuration and weights, and uses OpenCV to perform object detection.

### Credit

This application was developed by **ISAIAH OBOH**





