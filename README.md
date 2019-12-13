# ROB535Perception
Code for ROB535 Perception task 1/2

# Env Setup

The code is meant to run in the Google research colab environment. To do this, clone this repository into a google drive location. Note that this repo does not contain any of the data or initialized models due to size constraints. To obtain those, one must additionally download the following folders and extract them into the gDrive folder:

* data4Yolo - https://drive.google.com/open?id=1QM4Wg239epx3acTaNcBuIjJxpHJl0a75
* data - https://drive.google.com/open?id=1Kx1cPo5dCz82Rdqy4i4qj50KgJEetXq5

# Training the Model

To train the model, open the notebook 'Train_YOLOv3_ROB_535_UMICH.ipybn' in the colab environment, and run all cells. Further instructions are contained in the notebook if needed.

# Runing the Model

After running the 'Train_...' notebook, next open the 'Run_Yolo_3D_BoundingBox_UMICH.ipybn' notebook from colab and run all the cells.
