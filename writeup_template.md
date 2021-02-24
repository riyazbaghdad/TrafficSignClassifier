# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/training_images_graph.png "Visualization"
[image1.1]: ./output_images/augmented_training_images_graph.png "Visualization Augmented"

[image2]: output_images/0_resized.jpg "Grayscaling"
[image3]: output_images/1_resized.jpg "Random Noise"
[image4]: output_images/3_resized.jpg "Traffic Sign 1"
[image5]: output_images/4_resized.jpg "Traffic Sign 2"
[image6]: output_images/5_resized.jpg "Traffic Sign 3"
[image7]: output_images/yield_grey_resized.png "Traffic Sign 4"
[image8]: output_images/yield_grey_warped.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/riyazbaghdad/TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799 images
* The size of the validation set is 4410 images
* The size of test set is 12,630 images
* The shape of a traffic sign image is 32x32x3
* The number of unique classes in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)##

- convert the images to grayscale because of the computation cost factor involved when working on color RGB images, plus in this task the color has little significance.

- Saturating the intensity values at 1 and 99 percentile.
Min/max normalization to the [0, 1] range.
- Subtraction of its mean from the image, making the values centered around 0 and in the [-1, 1] range.
  
The percentile-based method gave an additional (approx.) 1% improvement over simple min/max normalization (this method was mentioned in the paper(Ciresan (2012): Multi-Column Deep Neural Network for Traffic Sign Classification)).

I considered the LeNet Network as my model classifier. I had to tweak the input dimensions to 32x32x1 as the input shape and increase the number of output feature maps as it improved my model accuracies. 

After several iterative approach, I felt that Dropout totally increased the robustness of the network thereby resulting in an increase with accuracy. Also by viewing the graph above of the training images, I decided to augment them to result in an increase of accuracy and robustness. 

Augmentation:
  - I tried several augmentation techniques like Rotation, Noise and Affine tranformation. After comparing the final training results, I had to stick with "Affine Transformation" as the augmentation resulted in an improvement.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3] ![alt text][image7]

Here is an example of an original image and an augmented image:

![alt text][image3] ![alt text][image8]

After Augmentation of training data, the size of augmented set is **69,598 images** 
- The following chart depicts the number of images per class;

![alt text][image1.1]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|      Layer      |                Description                 |
| :-------------: | :----------------------------------------: |
|      Input      |             32x32x1 Grayscale Image        |
| Convolution 5x5 | 1x1 stride, Valid padding, outputs 28x28x32|
|      RELU       |                                            |
|   Max pooling   | Kernel 2x2, stride 2x2, outputs 14x14x32   |
| Convolution 5x5 | 1x1 stride, Valid padding, outputs 10x10x64|
|      RELU       |                                            |
|   Max pooling   | Kernel 2x2, stride 2x2, outputs 5x5x64     |
|   Flatten       |                    1600                    |
|   Dropout       |                    70%                     |
|   FC Layer 1    |                    120                     |
|   RELU          |                                            |
|   Dropout       |                    70%                     |
|   FC Layer 2    |                    84                      |
|   RELU          |                                            |
|   Dropout       |                    70%                     |
|   FC Layer      |                    43                      |

 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Network Model  - LeNet Network
Optimizer      - ADAM
Batch size     - 128
No. of Epochs  - 20 (After an iterative approach)
Learning rate  - 0.01

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9 %
* validation set accuracy of 97.6 %
* test set accuracy of 95.70 %

 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6]

I chose 5 images to test on my model. All the images were good quality, however I felt like Image 4(Yield), Image 5(No passing) had some artifacts in it.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|        Image       |    Prediction    |
| :----------------: | :--------------: |
|  Stop Sign         |  Stop sign       |
|  Right of way      |  Right of way    |
|  Children crossing |  No passing      |
|  Yield             |  Yield           |
|  No passing        |  No passing      |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ~95%.    

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

**First Image : STOP sign**

![alt text][image2]

The top five soft max probabilities were;

| Probability |  Prediction   |
| :---------: | :-----------: |
|     .9970   |   Stop sign   |
|     .0010   |   Turn left ahead     |
|     .0003     |     Speed limit (120km/h)     |
|     .0003     | Speed limit (60km/h)   |
|     .0003   | Speed limit (70km/h) |

For the first image, the model is relatively sure that this is a 'stop sign'(probability of 0.99), and the image does contain a 'stop sign'. 

**Second Image : Right-of-way at the next intersection**

![alt text][image3]

The top five soft max probabilities were;

| Probability |  Prediction   |
| :---------: | :-----------: |
|     1.0000   |   Right-of-way at the next intersection   |
|     .0000   |   Beware of ice/snow     |
|     .0000   |     Double curve     |
|     .0000   | Children crossing   |
|     .0000   | Slippery road |

For the second image, the model is relatively sure that this is a 'Right-of-way at the next intersection (probability of 1.00)', and the image does contain a 'Right-of-way at the next intersection'. 

**Third Image : Children Crossing**

![alt text][image4]

The top five soft max probabilities were;

| Probability |  Prediction   |
| :---------: | :-----------: |
|     .4491   |   No passing   |
|     .2401   |   Dangerous curve to the left     |
|     .1047   |    No entry     |
|     .0874   | End of no passing   |
|     .0668   | Dangerous curve to the right |

For the second image, the model is relatively sure that this is a  'No passing' (probability of 1.00), and the image does not contain a 'No passing'. 

**Fourth Image : Yield**

![alt text][image5]

The top five soft max probabilities were;

| Probability |  Prediction   |
| :---------: | :-----------: |
|     1.0000   |   Yield  |
|     .0000   |   Priority road     |
|     .0000  |    No passing for vehicles over 3.5 metric tons |
|     .0000   | End of no passing   |
|     .0000   | No vehicles |

For the Fourth image, the model is relatively sure that this is a  'Yield' (probability of 1.00), and the image does contain a 'Yield'. 

**Fifth Image : No Passing**

![alt text][image6]

The top five soft max probabilities were;
    
| Probability |  Prediction   |
| :---------: | :-----------: |
|     1.0000   |   No passing  |
|     .0000   |   No passing for vehicles over 3.5 metric tons |
|     .0000  |    No entry |
|     .0000   | Vehicles over 3.5 metric tons prohibited   |
|     .0000   | No vehicles |

For the Fourth image, the model is relatively sure that this is a  'No Passing' (probability of 1.00), and the image does contain a 'No Passing'. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


