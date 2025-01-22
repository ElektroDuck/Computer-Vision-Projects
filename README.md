# Retail Item Recognition with Computer Vision
This project as been developed as assignements for the 2024 IPCV course

The project is dividend in 2 assignment covers 2 different approach of Computer Vision for the instance detection task:
- **Assignment 1**: Standard CV algoritm, use of sift keypoint
- **Assignment 2**: Deep learning, creation of CNN and finetune of the Resnet18

## Assignment 1

This project aims to develop a computer vision-based object detection system for supermarkets, designed to identify products on store shelves.
Given a set of products template the algorithm aims to identify this items on the store shelves.
The system will report the number, dimensions, and positions of each identified product.

![Image Example](res\task_1_multidetection.png)

### Proposed Solution: 

Cycle through the scene images taking the following steps:
1. **Denoise** the scene image
2. **Extract keypoints** from the denoised scene image
3. Detection cicle: 
    - **Compute the matches** between the scene keypoint and all the cached model keypoints
    - Considering only the model with more matches (best fitting model): 
    - **Compute the homography of the best fitting model** 
    - Controll if the detection is considered as a match checking if the **N_matching_kp > threshold**. The threshold can be a constant or calculated by an automated process that we designed (more about this later). 
    - If the model is considere as a match **add it to the report and delete the associated keypoint**. Removing the keypoint allow us to operate again the matching process without requiring masking out the product and reoperating the keypoint detection algorithm. It results in a significant drop of the time needed to analyze an image. 
    - If **no model detected the test fails the detection cicle terminates**.  
4. Draw boxes surrounding the products and add each image to an array of detected product

## Assignment 2
The goal of this assignment is to implement a neural network that classifies smartphone pictures of products found in grocery stores. 
The project is divided into two parts:
1. **Design a CNN From Scratch**: Design and implement a convolutional neural network from scratch for image classification, training it on the GroceryStoreDataset to achieve around 60% accuracy on the validation split.
2. **Finetune of the Resnet18**: Finetune the pretrained resnet-18 in order to reach an accuracy of at least 90%


### Dataset

![Task 2 Dataset](res\assignement_2_dataset.png)

The dataset you contains natural images of products taken with a smartphone camera in different grocery store. It includes products belong to the following 43 classes. For more information about the dataset check the relative [GitHub repository](https://github.com/marcusklasson/GroceryStoreDataset).

### Part 1 - Design a CNN From Scratch
For developing our network, we took inspiration from the VGG architecture. The final network consists of a stack of 5 modules, each composed of a convolution layer, batch normalization, ReLU activation, and a max-pooling layer. The model achieved an **accuracy of 0.66** on the test calibration set and 0.69 on the test set.

The accompanying notebook includes a detailed study explaining every component of the network and the choices made to achieve this architecture.

To train the CNN, we defined a training loop incorporating several key techniques: **AdamW optimizer** for adaptive learning rates and weight decay, and a training-validation process over multiple epochs. The **best model selection** is  based on the lowest validation loss, with **early stopping** to prevent overfitting. A StepLR **learning rate scheduler optimizes** training by adjusting the learning rate at specific intervals.

All the other training details are explained in detail in the notebook.
![Custom CNN Training](res\best_custom_cnn_training.png)

### Part2 - Finetune of the Resnet-18

For efficient fine-tuning, we used the same data preprocessing specified in the PyTorch ResNet-18 model pretrained on ImageNet-1K (V1) documentation. To avoid overfitting, we implemented data augmentation. Our training technique was divided into two parts: first, we froze the network parameters and trained only the classifier; in the second part, we trained all the layers of the network together. 

Thanks to this approach, we achieved a **91% accuracy** on the validation test and an 88.5% accuracy on the test set. All details related to the training are available and thoroughly described in the notebook.


![Resnet-18 Finetuning](res\best_resnet18_training.png)



