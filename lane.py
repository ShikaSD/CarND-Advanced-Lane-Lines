import cv2
import numpy as np

ym_per_pix = 30/720   # meters per pixel in y dimension
xm_per_pix = 3.7/700  # meters per pixel in x dimension


class Line:
    """
    Representation of one line of the Lane
    """
    def __init__(self, shape):
        """
        Initialize a line
        :param shape: shape of the image
        """
        # Bottom coordinate
        self.y = shape[0]

        # Current fit coefficient
        self.fit = None

        # List of recent fits
        self.__recent_fits = list()

        # Previous fit
        self.prev_fit = None

        # Maximum amount of fits to keep in __recent_fits
        self.__max_fits = 7

        # Amount of skipped frames
        self.not_detected_since = 0

    def bottom(self):
        """
        Get coordinate of a bottom point
        :return: current fit
        """
        if self.fit is None:
            raise ValueError("Fit is not defined")
        return self.fit[0] * self.y ** 2 + self.fit[1] * self.y + self.fit[2]

    def add_fit(self, fit):
        """
        Add fit to the recent fits
        :param fit: new fit to add
        """
        self.__recent_fits.append(fit)
        if len(self.__recent_fits) > self.__max_fits:
            self.__recent_fits = self.__recent_fits[1:]

    def clean_fits(self, amount):
        """
        Clean several fits from the beginning of recent fits
        :param amount: number of elements to remove
        """
        if len(self.__recent_fits) > amount:
            self.__recent_fits = self.__recent_fits[amount:]
        else:
            self.__recent_fits = []

    def mean_fits(self):
        """
        Get mean fit from recent ones
        :return: mean of the fits
        """
        return np.mean(np.array(self.__recent_fits), axis=0)


class Lane:
    """
    Representation of the lane
    """

    @classmethod
    def from_calib_images(cls, img_shape, calibration_img_names):
        """
        Create a lane with calibration parameters
        :param img_shape: shape of the frame
        :param calibration_img_names: names of calibration images
        :return: new instance of lane
        """
        mtx, dist = Lane.__create_undist_coeff(calibration_img_names)
        return Lane(img_shape, mtx, dist)

    def __init__(self, img_shape, mtx, dist):
        """
        Initialize a line
        :param img_shape: shape of the image
        :param mtx: camera matrix from calibration
        :param dist: distance coefficients from calibration
        """
        # Camera matrix
        self.mtx = mtx
        # Dist coeffs
        self.dist = dist

        # Shape of the frame
        self.shape = img_shape
        # Y coordinate span
        self.y_space = np.linspace(self.shape[0] // 2, self.shape[0] - 1, self.shape[0] // 2)

        src, dst = self.__get_warp_dims()
        # Matrix for warping
        self.M = cv2.getPerspectiveTransform(src, dst)
        # Inverted matrix for warping back
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        # Clahe object for normalization of brightness
        self.clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))

        # Left line
        self.left = Line(img_shape)
        # Right line
        self.right = Line(img_shape)

        # Margin for fit in __fit_lines and __fit_from_last
        self.fit_margin = 100
        # Maximum amount of frames to skip before starting searching from scratch
        self.skip_amount = 50

        # Width of the lane
        self.width = None

    @staticmethod
    def __create_undist_coeff(image_names):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        calib_img = None
        # Step through the image list and search for chessboard corners
        for fname in image_names:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if calib_img is None:
                calib_img = gray

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, calib_img.shape[::-1], None, None)
        return mtx, dist

    def __undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def __normalize(self, array):
        return np.uint8(255 * array / np.max(array))

    def __apply_threshold(self, src, mask, threshold):
        mask[(src >= threshold[0]) & (src <= threshold[1])] = 1
        return mask

    def __threshold(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=9)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=9)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        direction = np.arctan2(sobel_y, sobel_x)
        sobel_x = np.absolute(sobel_x)
        sobel_y = np.absolute(sobel_y)

        white = cv2.inRange(hls, (0, 230, 0), (255, 255, 255))
        yellow = cv2.inRange(hls, (0, 150, 150), (50, 255, 255))
        color_mask = cv2.bitwise_or(white, yellow)

        sobel_x_threshold = (40, 250)
        sobel_x = self.__normalize(sobel_x)
        sobel_x_mask = self.__apply_threshold(sobel_x, np.zeros_like(gray), sobel_x_threshold)

        sobel_y_threshold = (40, 250)
        sobel_y = self.__normalize(sobel_y)
        sobel_y_mask = self.__apply_threshold(sobel_y, np.zeros_like(gray), sobel_y_threshold)

        magnitude_threshold = (40, 100)
        magnitude = self.__normalize(magnitude)
        magnitude_mask = self.__apply_threshold(magnitude, np.zeros_like(gray), magnitude_threshold)

        direction_threshold = (0.7, 1.3)
        direction_mask = self.__apply_threshold(direction, np.zeros_like(gray), direction_threshold)

        result = np.zeros_like(gray)
        result[(color_mask == 255) | (sobel_x_mask == 1) & (sobel_y_mask == 1) | (direction_mask == 1) & (magnitude_mask == 1)] = 1

        return result

    def __get_warp_dims(self):
        w = self.shape[1]
        h = self.shape[0]
        src = np.float32([[200, h], [w // 2 - 30, h // 2 + 77], [w // 2 + 30, h // 2 + 77], [w - 150, h]])
        dst = np.float32([[300, h], [300, 0], [w - 300, 0], [w - 300, h]])

        return src, dst

    def __warp(self, img):
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def __fit_lines(self, src, line):
        histogram = np.sum(src[src.shape[0] // 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] / 2)
        if line == self.left:
            base = np.argmax(histogram[:midpoint])
        elif line == self.right:
            base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 15
        window_height = np.int(((src.shape[0]) // nwindows))
        nonzero  = src.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        current  = base

        minpix = 50

        lane_inds = []

        for window in range(nwindows):
            win_y_low  = src.shape[0] - (window + 1) * window_height
            win_y_high = src.shape[0] - window * window_height
            win_x_low  = current - self.fit_margin
            win_x_high = current + self.fit_margin

            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            lane_inds.append(good_inds)

            if len(good_inds) > minpix:
                current = np.int_(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds)

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        line.xpoints = x
        line.ypoints = y

        return np.polyfit(y, x, 2)

    def __fit_from_last(self, src, line):
        fit = line.fit

        nonzero = src.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        lane_inds = ((nonzerox > (fit[0] * (nonzeroy ** 2) + fit[1] * nonzeroy + fit[2] - self.fit_margin))
                     & (nonzerox < (fit[0] * (nonzeroy ** 2) + fit[1] * nonzeroy + fit[2] + self.fit_margin)))

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        line.xpoints = x
        line.ypoints = y

        return np.polyfit(y, x, 2)
    
    def __check_fits(self, line):
        sse_margin = 2000000

        if line.prev_fit is None:
            self.__detected(line)
            return

        fit_diff = line.fit - line.prev_fit
        error = fit_diff[0] * self.y_space ** 2 + fit_diff[1] * self.y_space + fit_diff[2]
        sse = np.sum(np.square(error))
        if sse > sse_margin:
            self.__repeat_last(line)
            return

        width = np.abs(self.right.bottom() - self.left.bottom())
        if not (self.width - self.fit_margin * 2 < width < self.width + self.fit_margin * 2):
            self.__repeat_last(line)
            return

        self.__detected(line)

    def __repeat_last(self, line):
        line.fit = line.prev_fit
        line.not_detected_since += 1

    def __detected(self, line):
        line.add_fit(line.fit)
        line.fit = line.mean_fits()
        line.not_detected_since = 0

    def __measure_curvature(self):
        # Calculate the new radii of curvature
        y_eval = self.shape[0]
        left_curverad = ((1 + (2 * self.left.fit[0] * y_eval + self.left.fit[1]) ** 2) ** 1.5) / np.absolute(2 * self.left.fit[0])
        right_curverad = ((1 + (2 * self.right.fit[0] * y_eval + self.right.fit[1]) ** 2) ** 1.5) / np.absolute(2 * self.right.fit[0])

        print(left_curverad, right_curverad)
        self.curvature = np.mean([left_curverad, right_curverad])

    def __print_summary(self, out):
        font = cv2.FONT_HERSHEY_SIMPLEX
        radius = "R = %.1fm" % self.curvature
        cv2.putText(out, radius, (20, out.shape[0] - 100), font, .5, (255, 255, 255), thickness=2)

        center = out.shape[1] // 2
        lane_center = (self.right.bottom() + self.left.bottom()) / 2
        dist = "Distance = %.1fm" % (np.abs(lane_center - center) * xm_per_pix)
        cv2.putText(out, dist, (20, out.shape[0] - 70), font, .5, (255, 255, 255), thickness=2)

        return out

    def draw_lane(self, img):
        """
        Draws lane on the img provided
        :param img: frame to find lane on
        :return: frame with drawn lane
        """
        src = self.__undistort(img)

        # Preprocess image in lab space; fix lightness
        lab_image = cv2.cvtColor(src, cv2.COLOR_RGB2LAB)
        lab_image[:, :, 0] = self.clahe.apply(lab_image[:, :, 0])
        img = cv2.cvtColor(lab_image, cv2.COLOR_Lab2RGB)

        # Threshold and warp an image
        warped = self.__warp(self.__threshold(img))

        # Fit left line
        if self.left.fit is None or self.left.not_detected_since > self.skip_amount:
            self.left.fit = self.__fit_lines(warped, self.left)
        else:
            self.left.fit = self.__fit_from_last(warped, self.left)

        # Fit right line
        if self.right.fit is None or self.right.not_detected_since > self.skip_amount:
            self.right.fit = self.__fit_lines(warped, self.right)
        else:
            self.right.fit = self.__fit_from_last(warped, self.right)

        # Check validity of the fits
        self.__check_fits(self.left)
        self.__check_fits(self.right)

        # Measure curvature and write it to a image
        self.__measure_curvature()
        summary = self.__print_summary(np.zeros_like(img))

        # Set width if not set
        if self.width is None:
            self.width = np.abs(self.right.bottom() - self.left.bottom())

        # Draw the lane
        left = (self.left.fit[0] * self.y_space ** 2 + self.left.fit[1] * self.y_space + self.left.fit[2])
        right = (self.right.fit[0] * self.y_space ** 2 + self.right.fit[1] * self.y_space + self.right.fit[2])

        pts_left = np.array([np.stack([left, self.y_space], axis=1)])
        pts_right = np.array([np.flipud(np.stack([right, self.y_space], axis=1))])

        pts_left = cv2.perspectiveTransform(pts_left, self.Minv).astype(int)
        pts_right = cv2.perspectiveTransform(pts_right, self.Minv).astype(int)

        lines = np.zeros_like(src)
        cv2.polylines(lines, pts_left, False, (255, 0, 0), 10)
        cv2.polylines(lines, pts_right, False, (255, 0, 0), 10)
        cv2.fillPoly(lines, np.hstack([pts_left, pts_right]), (0, 255, 0))

        # Combine the layers
        img = cv2.addWeighted(src, 1, lines, .4, 0.)
        img = cv2.addWeighted(img, 1, summary, .4, 0)

        return img
