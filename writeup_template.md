# **Traffic Sign Recognition** 

## Writeup

The aim of this project is to develop an algorithm using deep learning convolution neural network to classify traffic signs. The project invols the following steps.


The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./BarChart.png "Class Dataset Barchart"
[image2]: ./TrafficSigns.png "Traffic Signs"
[image3]: ./mean_variance.png "Signal Conditioning"
[image4]: ./MinMaxNorm.PNG "Min Max Normalization"
[image5]: ./Normalization.png "Before and after normalization"
[image6]: ./LeNet.png "LeNet Architecture"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

### Rubric Points

## Here the project reqirement [rubric points](https://review.udacity.com/#!/rubrics/481/view) are considered individually and described in detail how the implementation is done.

---
### Project Files

The project files includes the python project code written in jupyter notebook, input dataset stored in .picle format, traffic sign pictures downloaded from internet.
 * Here is a link to the [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Dataset Summary
The input data contains three sets of data called the training, validation and test. Only the training dataset is used for training the neural network. The proformance of trained network is validated using the validation dataset. Based on the performance of the network measured from the training and validation set the parameters of the network are tuned to acheive maximum possible performance. In the following sections it is explained in detail how the parameters of the network are tuned. Finally after tuning of parameters the network is tested with test dataset. 

The summary of the dataset are calculated using python native features and then the dataset is visualized using matplotlib library. The basic summary of the dataset follows:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.
The images in the dataset are grouped according its class and the no of images in each class are plotted as a bar chart in the following picture.


![alt text][image1]

The firt thing that can be noted is that the number of images in the training dataset are greater than validation and test dataset. Large amount of training data is useful because the network learns better when it is trained with large dataset. It should also be noted that the classes such as 'speed limit 50 km/hr' have large amount of data compared to class such as "Dangerous Curve to the Left". The network might have difficulty in classifying the traffic signs which have less data. In such cases it is better to augument the data for such classes. But in this project this is not done as the network acheived satisfactory performance.

The datasets have 43 different traffic signs one image in each of the traffic signs are shown in the picture below.


![alt text][image2]

### Design and Test a Model Architecture

#### 1. Preprocessing

A raw image in RGB format is represented in form of an array of size m x n x 3 where is m,n are number of pixes along width and height of the image and 3 represents three channels red,blue and green. Each value in this array ranges from 0 to 255 for an 8 bit image. The gradient desent algorithm works better if the input dataset is normalized i.e. the image has equal variances in both horzontal and vertical directions and zero mean. 

![alt text][image3]

In this project a simpler approach called min-max scaling is used. This approach scales the input values from 0 to 255 in a 8 bit image to a values in a given range. The following formula is used to do min-max scaling.

Min-Max Scaling:

![alt text][image4]

In the above forumla a and b are choosen output range chosen in the project as 0 and 1. X_min and X_max are the minimum and maximum values of the input data. This formula it simpler because the mean and variance are avoided and it acheives good performance. The picture below shows the images before and after normalization.

![alt text][image5]

This project feeds RGB coloured image to the network as the convolution neural networks are able to handle 3 channel images and also the colour information can be useful in classifying the traffic signs as some signs are in different colours.

#### 2. Model Architecture

The LeNet architecture is used in this project. The LeNet architecture is a multi layed neural network architecture which is popularly used to classify handwritten character recognition. The LeNet architecture contains two sets of convolution layer and maximul pooling layer alternatevely followed by three fully connected layer.The convolution and fully connected layer uses rectified logic units(Relu) for activation.Finally the output of the final is feed to softmax function to calculate the probablities of the classes. The following image and the table shows the LeNet architecture used in this project.


![alt text][image6]
source: Yann Lecunn

| Layer         		|     Description	        					| 
|:---------------------|:---------------------------------------------| 
| Input         		| 32x32x3 RGB image   							| 
| 1st Convolution	| 5x5x3x6 filter, 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|Activation											|
| Max pooling	    | 2x2 filter, 2x2 stride,  outputs 14x14x6 				|
| 2nd Convolution | 5x5x6x16 filter, 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|Activation												|
| Max pooling	    | 2x2 filter, 2x2 stride,  outputs 5x5x16 				|
|Flatten| Reshapes the input array into a vector/row matrix,output 5x5x16 = 400 |
| Fully connected		|Flattened array is connected to 120 hidden nodes,  weights 400x120,bias 120			|
| RELU					|Activation												|
| Fully connected		|Hidden layer 84 nodes,  weights 120x84,bias 84		|
| RELU					|Activation			|
| Fully connected		|Output layer nodes = no. of classes(43),  weights 84x43,bias 43		|
| Softmax				| Activation to calulate probablities of the classes							| 


#### 3. Model Training

The model is trained using gradient decent approach to minimize the loss

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


