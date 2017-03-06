#Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted.png "Undistorted"
[image2]: ./output_images/pipeline_undistorted.png "Road Transformed"
[image3]: ./output_images/binary_example.png "Binary Example"
[image4]: ./output_images/warped_lines.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file `camera_cal.py`. The calibration consists in establishing a
transformation between points in real world space (object points) and points in image space (image points). Object
points are assumed to follow a known pattern described in a coordinate system fixed to a chessboard image. Image points
are obtained using the function `cv2.findChessboardCorners` (line 32) with a given pattern. Since the pattern may be different
among images, the script looks for some possible patterns in order to use more information for calibration. For more
precise image points the script uses `cv2.cornerSubPix` (line 49).

The function `cv2.calibrateCamera` (line 55) then uses objects and image points to obtain the camera calibration matrix and the
distortion coefficients. The calibration data is stored in the file `cal_data.p` (line 68) for future use.

This is how a sample image looks like before and after distortion correction using the calibration data and
the `cv2.undistort` function:

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
Given the calibration matrix as `mtx` and distortion coefficients as `dist`, a corrected image can be obtained as the
output of `cv2.undistort(img, mtx, dist)`, where `img` is the original distorted image. Here is a sample result:

![alt text][image2]

The effect of distortion is more noticeable at the periphery of the image.

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used the HSV color space and Contrast Limited Adaptive Equalization (CLAHE) on the V channel in order to get a binary
image containing mostly the lane lines. The implementation is the `hsv_v_thresh` function (lines 45 - 57) in the file
`color_spaces.py`. The function is used in the `apply_thresholds` method of the `LaneFinder` class defined in the file
`lane_finder.py` (lines 290 - 291). The file `color_spaces.py` contains tests using alternative approaches. Below is a example of a binary image result.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image`, which appears in lines 7 through 28 in
the file `image_warping.py`. The `warp_image` function takes as input a binary image (`bin_img`). Source (`src`) and
destination (`dst`) points are hardcoded in the function definition.  I chose to hardcode the source and destination
points in the following manner (lines 13 - 21):

```
src = np.float32([[(w / 2) - 60, h / 2 + 100],
                  [((w / 6) - 10), h],
                  [(w * 5 / 6) + 10, h],
                  [(w / 2 + 60), h / 2 + 100]])

dst = np.float32([[(w / 4), 0],
                  [(w / 4), h],
                  [(w * 3 / 4), h],
                  [(w * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1076, 720     | 960, 720      |
| 700, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial.

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

