First project was designed to get back on track with more progress with knowledge.
I took AI tools to help me get to https://www.kaggle.com/datasets/ultralytics/brain-tumor where I took their files

Files consisted for YOLOv8 data train + ultralytics where we have labels and images
(labels have 0-negative 1-positive image with quadrant around tumor coordinates)
With setting everything I managed to create a code with tensorflow and pytorch libraries

First I needed to separate the images to positive and negative folders creating Sort_PN to new folder.
Changing the paths for train then validate

WIth that folder I will train the model (naming him tumor_model.h5) - in figure1 we can see the results.

Requirements:

tensorflow
numpy
matplotlib
torch
torchvision
numpy
matplotlib

Optional 

scikit-learn
pandas
opencv
ultralytics

TensorFlow: 2.14.0
PyTorch: 2.8.0+cpu
NumPy: 1.26.0
Matplotlib: 3.8.0
Pillow: 8.4.0
