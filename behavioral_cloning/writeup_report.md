#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use a simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.png "Model Visualization"
[image2]: ./images/center.jpg "Center lane driving sample"
[image3]: ./images/center_2017_02_14_20_56_28_982.jpg "Recovery Image"
[image4]: ./images/center_2017_02_14_20_56_29_658.jpg "Recovery Image"
[image5]: ./images/center_2017_02_14_20_56_30_339.jpg "Recovery Image"
[image6]: ./images/center.jpg "Normal Image"
[image7]: ./images/center_flipped.jpg "Flipped Image"

## Rubric Points

###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 run_dir
```
Where <code>run_dir</code> is the directory where the frames will be saved.
####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model is similar to the NVIDIA's model described in this [paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), but using a smaller number
of parameters in order to decrease the training time.

It includes 5 convolutional layers with 18, 24, 36, 48 and 48 output filters, respectively. The first 3
layers use 5x5 filters and the last 2 layers use 3x3 filters. All the convolutional layers are
followed by ReLU activations.

There are max pooling layers with a 2x2 pool size following the first 3 convolutional layers and, in order to
decrease overfitting, dropout layers with a 0.2 drop probability following the last 2 convolutional layers.

The final part consists of 5 fully connected layers with output dimensions 1064, 100, 50, 10 and 1. There are hyperbolic
tangent activations after the first 4 fully connected layers.

Before the convolutional layers the input images are cropped
and normalized using Cropping2D and Lambda layers, respectively. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 108 and 111). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 121).
The [Adam](https://arxiv.org/abs/1412.6980) optimizer is known to require little or no tuning.
####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving for about
two laps and additional data for curves and some recoveries from the left and right sides.

The recovery problem was also addressed with images from the left and right cameras and adjusted steering angles during
the training phase.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to generate some training/validation/testing data and evaluate
the performance of proposed models with increasing complexity.

My first step was to use a simple 2 layer convolutional model, which didn't have enough capacity to keep the vehicle on
the track. Then I used a VGG-like model without much success.

Finally, I used a model similar to the NVIDIA's model, but with a smaller number of parameters. I thought this model
might be appropriate because it was successfully used in a much more challenging instance of the autonomous driving problem,
including diverse weather conditions and road obstacles. The reduced number of parameters helped to increase training speed
and was consistent with the relative simplicity of the problem to be solved.

In order to gauge how well the model was working, I split my image and steering angle data into a training, a validation
 and a test set. The test set was used to further evaluate the model on completely unseen data
 after the last training session in order to detect overfitting.

To combat the overfitting, I inserted two dropout layers after the last two convolutional layers of the model.

Then I tuned the dropout probability and the correction applied the images of the left and right cameras. Flipped images
of the center camera were used to augment the training data in real time using a generator.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots
where the vehicle fell off the track, mainly sharp curves. To improve the driving behavior in these cases, I collected
more data during curves and some data during recovering from the margins of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 87-119) consisted of a convolution neural network with 5 convolutional
layers and 5 fully connected layers. Max pooling was used after the first 3 convolutional layers and dropout after the
last two.

Here is a visualization of the architecture generated using the keras.utils.visualize_util module:

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle
would learn appropriate steering angles to return to center lane driving. These images show what a recovery looks like
starting from the left side of the road:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data set, I also flipped the center camera images and angles thinking that this would provide additional useful information
to the training algorithm. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

The data augmentation was performed in real-time using a Python generator and the appropriate Keras methods.

After the collection process, I had 17373 data points, not including the augmented data. The data was processed by cropping
the region of interest corresponding to the road and scaling each pixel to the range [-0.5, 0.5].

I finally randomly shuffled the data set and put 20% of the data into a validation set. More 20% of that training dataset
was reserved as a test data set, to be used after the whole training process.

I used this training data for training the model. The validation set and the test set helped determine if the model was
over or under fitting. The ideal number of epochs was 5 as evidenced by the changes in the validation and test accuracies.
I used an Adam optimizer so that manually training the learning rate wasn't necessary.
