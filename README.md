# ID Card Segmentation using PyTorch & OpenCV
A Deep Learning–based Image Segmentation Project to identify an ID Card on an image.  

### Objectives
The goal of this project is to recognize a ID Card on a photo, cut it out using semantic segmentation and to 
transform the perspective so that you get a frontal view of the ID Card.

## Problem Statement 
In many real-world scenarios, ID cards are photographed with cluttered backgrounds, varying lighting, or partial occlusions.
This makes it difficult for OCR systems or verification platforms to process them accurately.

Goal:
Build a system that accurately segments ID cards from an image so they can be processed further.

## Technologies used
Language - Python
Deep Learning - PyTorch
Computer Vision - OpenCV



## Additional Information
Dataset: [MIDV-500](https://arxiv.org/abs/1807.05786)  

Trained on a NVIDIA GeForce RTX 3090

## Installation
1. Create and activate a new environment.
```
conda create -n idcard python=3.9.1
source activate idcard
```
2. Install Dependencies.
```
pip install -r requirements.txt
```

## Download and Prepare Dataset
Download the image files (image and ground_truth).  
Splits the data into training, test and validation data.
```
python prepare_dataset.py
```

### Training of the neural network
```
python train.py --resumeTraining=True
```
`resumeTraining` is optional an resumes training on an existing `./pretrained/model_checkpoint.pt`

### Test the trained model
```
python test.py test/sample1.png --output_mask=test/output_mask.png --output_prediction=test/output_pred.png --model=./pretrained/model_final.pt
```

Call `python test.py --help` for possible arguments. 

### Additional commands
Starts Tensorboard Visualisation.
```
tensorboard --logdir=logs/
```
## Project Structure 
ML_IDCard_Segmentation_Pytorch/
│
├── pretrained/             Pretrained segmentation model
├── data/                   Sample images
├── utils/                  Helper scripts
├── models/                 Network architecture
├── inference.py            Run model inference
├── train.py                Training script (if required)
├── README.md               Documentation
└── requirements.txt        Required packages

## Background Information

### Model
A [U-NET](https://arxiv.org/abs/1505.04597) was used as the model.
U-Net is a convolutional neural network that was developed for biomedical image segmentation at the
Computer Science Department of the University of Freiburg, Germany.
The network is based on the fully convolutional networkand its architecture was modified and extended to work with
fewer training images and to yield more precise segmentations. 
Segmentation of a 512*512 image takes less than a second on a modern GPU.
  
![IoU](assets/unet.jpg "U-Net")

### Metrics
The Metric [IoU](https://arxiv.org/abs/1902.09630) (Intersection over Unit / Jaccard-Coefficient) was used
to measure the quality of the model.
The closer the Jaccard coefficient is to 1, the greater the similarity of the quantities. The minimum value of the Jaccard coefficient is 0.   
![IoU](assets/iou_formular1.png "IoU")
  
Example:  
![IoU](assets/iou.png "IoU")

## Results for validation set (trained on the complete dataset)
Intersection over Unit:  
0.9939

## Conclusion 
This project successfully demonstrates how deep learning can be used to perform accurate document segmentation.
By isolating the ID card from any background, the system enhances the accuracy of downstream tasks such as OCR, verification, and classification.

## Author
Prerna 

