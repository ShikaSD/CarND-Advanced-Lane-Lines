
## Advanced Lane Finding Project**

[//]: # (Image References)

[calibration]: ./output/calib0.jpg "Undistorted"
[dist-corrected]: ./output/test0.jpg "Road Transformed"
[threshold]: ./output/threshold.jpg "Binary Example"
[warped]: ./output/warped.jpg "Warp Example"
[fit]: ./output/fit.jpg "Fit Visual"
[lane]: ./output/lane.jpg "Output"
[video1]: ./project_video_result.mp4 "Video"
[notebook]: ./lane-lines.ipynb "notebook"
[source]: ./lane.py "source"

### Camera Calibration

#### 1. Distortion correction

To compute the calibration coefficients, I have used cv2 methods for chessboard detection. First, I have used provided calibration images with the chessboard to detect corners and add use them as the `imgpoints` while using hardcoded `objpoints`.

Then, `cv2.calibrateCamera` method is used to retrieve camera matrix and distance coefficients. They are used in `cv2.undistort` method to undistort the image with given calibration parameters.

The example of the correction can be seen in the [notebook] in first code cell or in the [source] in lines #125 - #155.

**Undistorted calibration image**:

![alt text][calibration]


### Pipeline for single images (In the [notebook])

#### 1. Distortion corrected image.

One of the test images with corrected distortion

![alt text][dist-corrected]

#### 2. Thresholding

I used several techniques to select valid regions of the images. First, the images were corrected with a CLAHE algorithm on lightness plane of LAB image. Later, the image was thresholded with Sobel filter in X and Y directions, white and yellow color selection, Sobel magnitude and filter direction.
Combining them together, I received the following image from the one above.

![alt text][threshold]

#### 3. Perspective transform

For the warping perspective, I have set the following points:

```python
src = np.float32([[200, h], [w // 2 - 30, h // 2 + 77], [w // 2 + 30, h // 2 + 77], [w - 150, h]])
dst = np.float32([[300, h], [300, 0], [w - 300, 0], [w - 300, h]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 200, 720      | 300, 720      |
| 610, 437      | 300, 0        |
| 670, 437      | 980, 0        |
| 1130, 720     | 980, 720      |

I verified that the transform is working as expected warping the image and checking that the lines are roughly parallel.

![alt text][warped]

#### 4. Identification of lane pixels and fitting a polynomial

To fit the lines, first I have chosen two peaks of on the histogram of bottom half of the image to find the peaks. Then using the windows, I have found the, stacking windows vertically and moving the center of the upper window in general direction of current one.

The example can be seen on the following image. Here the points of each line are red and blue while polynomial is fit to those points.

![alt text][fit]

#### 5. The radius of the curvature

I used the formula from the course to calculate it with given left and right fit.
Finally, I've received the following snippet to use:
```python
y_eval = self.shape[0]
left_curverad = (
  (1 + (2 * self.left.fit[0] * y_eval * ym_per_pix + self.left.fit[1]) ** 2) ** 1.5
) / np.absolute(2 * self.left.fit[0])
right_curverad = (
  (1 + (2 * self.right.fit[0] * y_eval * ym_per_pix + self.right.fit[1]) ** 2) ** 1.5
) / np.absolute(2 * self.right.fit[0])

curvature = np.mean([left_curverad, right_curverad])
```

#### 6. The example

The last cell in the notebook is showing the code to plot the lane onto the image. To achieve this, I have used linspace to plot the points of the fitted line and later used `cv2.perspectiveTransform` to transform resulting points to the image space.

![alt text][lane]

---

### Video

Here's a [link to my video result](./project_video_result.mp4)

---

### Issues

The pipeline is doing reasonably well while dealing with the project video, especially with taking into consideration fitting to mean value of the previous fit values and skipping frames while dealing with wrong fit values.

However, the there are many issues that are not fixed in this pipeline. As it can be seen in challenge videos, the lane is not always detected based on right line positions. Moreover, lightness fix doesn't help much when the line is completely hidden because of the sun.

Furthermore, the pipeline is not doing well in the environment where horizon is a little bit higher or lower. Therefore, dynamic positioning of the lane boundaries could be implemented here.
