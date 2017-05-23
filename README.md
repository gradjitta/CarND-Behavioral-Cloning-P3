# **Behavioral Cloning Project**

---


[//]: # (Image References)

[image6]: ./examples/flip.png "Normal Image"
[image7]: ./examples/augment1.png "Data left and right camera"
[image8]: ./examples/valid.png "Training plot"
[image9]:  ./examples/shadow.jpg "Shadow Image"
[image10]: ./examples/bridge.jpg "Bridge Image"
[image11]: ./examples/steep1.jpg "Steep turn 1 Image"
[image12]: ./examples/steep2.jpg "Steep turn 2 Image"

##### I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (modifed version)
* model.h5 containing a trained convolution neural network (that worked)
* README.md  writeup of the project
* video.mp4  The final video generated from the trained model ``model.h5``


##### 2. Submission includes functional code

The model.py has data augmentation and the Nvidia model, as functions.

Using the ``drive.py`` and the model generated (``model.h5``), executing the following:
```sh
python drive.py model.h5
```
results in successfully navigating the track 1.

##### 3. Submission code is usable and readable
#### `model.py`

The model.py file contains the code for training and saving the convolution neural network. Furthermore, the code is readable with required comments.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The input image size considered was 66x200, after required cropping (to focus on the road ) and resizing the original image. The model uses Nvidia model as described in [the paper](https://arxiv.org/pdf/1604.07316.pdf). It consists of a convolution neural network with 9 layers (5 convolutional layers and 3 fully connected layers). The first 3 convolutional layers use strided 5x5 kernels followed by the convolutional layers that use non-strided 3x3 filters. I picked an already working model architecture and tried to train a model that navigates the entire track and building a good architecture from scratch would require lot of experimentation.

Layer (type)                     |Output Shape          |Param #     |Connected to                     
--- | --- | --- | ---
lambda_1 (Lambda)                |(None, 66, 200, 3)  |0       |lambda_input_1[0][0]
convolution2d_1 (Convolution2D)  |(None, 31, 98, 24)  |1824    |lambda_1[0][0]
convolution2d_2 (Convolution2D)  |(None, 14, 47, 36)  |21636    |convolution2d_1[0][0]   
convolution2d_3 (Convolution2D)  |(None,  5, 22, 48)  |43248    |convolution2d_2[0][0]   
convolution2d_4 (Convolution2D)  |(None,  3, 20, 64)  |27712    |convolution2d_3[0][0]   
convolution2d_5 (Convolution2D)  |(None,  1, 18, 64)  |36928    |convolution2d_4[0][0]
flatten_2 (Flatten)              |(None, 1152)        |  0      |     convolution2d_5[0][0]           
dense_6 (Dense)                  |(None, 1164)        |  1342092|     flatten_2[0][0]                  
activation_5 (Activation)        |(None, 1164)        |  0      |     dense_6[0][0]                    
dropout_4 (Dropout)              |(None, 1164)        |  0      |     activation_5[0][0]               
dense_7 (Dense)                  |(None, 100)         |  116500 |     dropout_4[0][0]                  
activation_6 (Activation)        |(None, 100)         |  0      |     dense_7[0][0]                    
dense_8 (Dense)                  |(None, 50)          |  5050   |     activation_6[0][0]               
activation_7 (Activation)        |(None, 50)          |  0      |     dense_8[0][0]                    
dense_9 (Dense)                  |(None, 10)          |  510    |     activation_7[0][0]               
activation_8 (Activation)        |(None, 10)          |  0      |     dense_9[0][0]                    
dense_10 (Dense)                 |(None, 1)           |  11     |     activation_8[0][0]  
Total params : 1,595,511|||


#### 2. Attempts to reduce overfitting in the model

Different dropout layers were considered to reduce over-fitting. In the code ``model.py``, we added a dropout layer of 0.2 in line 100. We could use different combinations of these layers across the architecture. The choice used in the code worked to navigate the track 1 completely.

#### 3. Model parameter tuning

The model used an Adam optimizer so tuning its learning rate is done implicitly. The choice of using Adam is quite prevalent in the convolution neural network literature.

#### 4. Appropriate training data

The training data used:
* Dataset provided by Udacity
* Dataset from driving in center lane
* Dataset for recovery from road edges

### Data collection and Training


I first used the dataset provided by Udacity and the Nvidia model to see where
the model fails in general.  The following are the places where the model fails:

1) Shadow

![alt text][image9]

2) Bridge

![alt text][image10]

3) Steep turn left

![alt text][image11]

4) Steep turn right

![alt text][image12]

Next, I added more data, as suggested in the course tips, by recording 2 laps of center lane driving. This should help in solving the problems faced by the shadow and steep turns. In order to make the car navigate successfully on the bridge, I added another dataset where I try to recover the car driving towards the rails. This needs to be done since the model learns to only go straight on the bridge. Using this recovery data, it helped to successfully navigate the bridge. After adding all the images, I had around 12791 images. Next, I used these images to generate more artificial/augmented data.

For data augmentation, the images were fliped and its corresponding seering angles were modified accordingly.

![alt text][image6]

Further data augmentation is done on the dataset from left and right cameras. The figures below are from center, left, and right cameras (the figures are shown in RGB, but in reality the YUV images are fed in every batch).


![alt text][image7]


Finally, the dataset I used was around 12791 x 6. Preprocessing and shuffling of these images was done within the python generator function. I cropped the images along its height (top 60 and bottom 25 pixels), followed by resizing the image to 200 x 66
(choice of using YUV and 200 x 66 images is from the Nvidia paper)

The dataset training/validation split I used is 80/20. 

#### Training 
Inorder to get the final model that was successful, I ran the training for 5 epochs. The plot of training and validation errors vs epochs is shown below.

![alt text][image8]


#### Video
Also find video.mp4 file in the repository.
