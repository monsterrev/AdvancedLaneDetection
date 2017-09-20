## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

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

[image1]: ./output_images/originalChessboard.png "Original"
[image2]: ./output_images/unDistortedChessboard.png "Undistorted"
[image3]: ./output_images/Original.png "Original"
[image4]: ./output_images/Undistorted.png "Undistorted"
[image5]: ./output_images/OriginalThreshold.png "OriginalThreshold"
[image6]: ./output_images/BinaryThreshold.png "BinaryThreshold"
[image7]: ./output_images/PT_Original.png "Original"
[image8]: ./output_images/Warped.png "Warped"
[image9]: ./output_images/Undistorted-Thresholded-Warped.png "Undistorted-Thresholded-Warped"
[image10]: ./output_images/lanePixels.png "LanePixelswithBoundaries"
[image11]: ./output_images/laneDrawn.png "LaneMarked"
[image12]: ./output_images/MarkedRC.png "Radius Of Curvature and Distance from Center"
[image13]: ./output_images/OriginalStraightLine.png "OriginalStraight"
[image14]: ./output_images/WarpedStraightLine.png "WarpedStraightLine"
[video1]: ./result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "AdvancedLaneDetection.ipynb" (lines 8 through 38)

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. The openCV function findChessboardCorners is used to detect the obj_points, all the images under calibration folder are fed to the function after converting them to gray scale. Returned obj_points is appended to objPoints and `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1] ![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3] ![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and sobel gradient thresholds to generate a binary image (thresholding steps in "AdvancedLaneDetection.ipynb" under ThresholdedBinaryImage section).  L channel of HLS helped to isolate white lights and S channel to isolate yellow. System is supposed to be robust to varying lighting conditions and for the same reason max threshold values are chosen to be 255. Absolute Scaled Sobel helped with x derivative to accentuate lines away from horizontal. Combination of color and gradient thresholds was used to generate the binary image.
Here's an example of my output for this step.  

![alt text][image5] ![alt text][image6]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `unwarp()`, which appears in lines 1 through 10 in the file `AdvancedLaneDetection.ipynb` under PerspectiveTransform section.  The `unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32([(400,0),
                  (w-400,0),
                  (400,h),
                  (w-400,h)])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 560,460       | 320, 0        | 
| 700,460       | 320, 720      |
| 400,670       | 960, 720      |
| 950, 670      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Here is an example of test image and its perspective transform..

![alt text][image7] ![alt text][image8]

This is how it performs on StraightLine Images
![alt text][image13] ![alt text][image14]

Then I did Perspective Transform on Undistorted BinaryThresholded Image..
![alt text][image7] ![alt text][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Code is in the jupyter Notebook under "Detect lane pixels and fit to find the lane boundary" section. The function slidingWindowLineDetection() is inspired from the classroom , and helps identify lane lines and fit the second order polynomial to both right and left lane lines.  First I plotted a histogram and based on local maxima of left half and right half of the histogram and chose the points which marks the bottom most portion of both left and right lines. Then I followed the sliding window protocol to identify lane pixels,  each one centered on the midpoint of the pixels from the window below. Number of windows chosen were 10 which typically makes us follow the lane lines upto the top of the binary image.

![alt text][image10]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in my code in jupyter notebook `AdvancedLaneDetection` under the section "Radius of Curvature and Position of Vehicle wrt center". Radius of curvature is at any point x of function f(y) is defined as 
```python
Rcurve=∣d2y/d2x∣/[1+(dy/dx)2]3/2
​
```
The y values of your image increase from top to bottom, so if, for example, you wanted to measure the radius of curvature closest to your vehicle, you could evaluate the formula above at the y value corresponding to the bottom of your image, or in Python, at yvalue = image.shape[0]

Equation : # Calculate the new radii of curvature
```python
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

```

The position of vehicle wrt center of lane is given
```python
lane_center_position = (r_fit_x_int + l_fit_x_int) /2
center_dist = (car_position - lane_center_position) * xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in second cell of section "Radius of Curvature and Position of Vehicle wrt Center" in function "drawBackLane". This steps involves generating a polygon based on left_fit and right_fit values obtained from slidingWindowLineDetection, this is warped back on the original image using Minverse (InversePerspective) and superimposed on original image.
 Here is an example of my result on a test image:

![alt text][image11]

Example with marked Radius of Curvature and Distance from center.

![alt text][image12]
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

My current approach runs the sliding window algorithm for each frame of the video and does not pick up the points determined from the previous frame. I tried to follow the approach as described in the class where we save the previous leftfit and right fit values in Line class and use the function UsingLastFitData. But I could not capitalize on it and my approach failed and main reason for failure was with saving the fit values, I was trying to average out the values and this was leading to detection of lane lines at the center of the road in between the lane lines (i.e. middle of nowhere). I dont necessarily understand the proper reason for this but my theory says its because of the averaging done but then I could not think of any other approach. 
Next challenge was to deal with varying lighting conditions, but then using the LChannel and SChannel helped. To think about different color spaces was the key. The white lines were detected with the L channel but there might be problems when white line would not contrast with rest of image i.e. whenever there is snow fall.

The pipeline does not effectively works whenever there is steep change in lane directions, e.g road is curving right, till now things would be fine and all of a sudden there is steep left curve, the lane boundaries go bit outside of the lanes which I believe is not that bad but yeah can be improved. One probable solution would be to average out the fits or include a confidence level threshold for the fits and rejecting them if there is too much deviation.
