import cv2
import numpy as np
import glob
import os
import pickle

# Define the class object to store and process the characteristics of each line detected, and the processing pipeline
class Line():
	def __init__(self):
		# Was the line detected in the last iteration?
		self.detected = False
		# Lane polynominal coefficients
		self.left_fit = None
		self.right_fit = None
		# Lane averaging buffer number
		self.average_n = 12
		# Averaging lanes container
		self.left_container = np.zeros((self.average_n, 720))
		self.right_container = np.zeros((self.average_n, 720))
		# Current image processing iteration counter
		self.cursor = 0

	def basic_lane_find(self, binary_warped):
		"""
		Implement sliding windows and fit a polynomial
		"""
		# Take a histogram of the bottom half of the image
		histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :, 0], axis=0)
		# Find the peak of the left and right halves of the histogram
		midpoint = np.int(histogram.shape[0] / 2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint
		# Choose the number of sliding windows
		nwindows = 9
		# Set height of windows
		window_height = np.int(binary_warped.shape[0] / nwindows)
		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Current positions to be updated for each window
		leftx_current = leftx_base
		rightx_current = rightx_base
		# Set the width of the windows +/- margin
		margin = 100
		# Set the minimum number of pixels found to recenter window
		minpix = 50
		# Create empty lists to receive left and right lane pixel indices
		left_lane_inds = []
		right_lane_inds = []
		# Step through the windows one by one
		for window in range(nwindows):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = binary_warped.shape[0] - (window + 1) * window_height
			win_y_high = binary_warped.shape[0] - window * window_height
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin
			# Identify the nonzero pixels in x and y within the window
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
				nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
				nonzerox < win_xright_high)).nonzero()[0]
			# Append these indices to the lists
			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)
			# If you found > minpix pixels, recenter next window on their mean position
			if len(good_left_inds) > minpix:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > minpix:
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
		# Concatenate the arrays of indices
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)
		# Extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds]
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]
		# Fit a second order polynomial to each
		self.left_fit = np.polyfit(lefty, leftx, 2)
		self.right_fit = np.polyfit(righty, rightx, 2)
		#Generate x and y values for plotting
		ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
		left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
		right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]
		self.detected = True

		return left_fitx, right_fitx

	def lane_find(self, binary_warped):
		"""
		Skip the sliding windows step once lines found
		"""
		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Set th ewidth of the windows +/- margin
		margin = 100
		left_lane_inds = ((nonzerox > (self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] - margin)) & 
			(nonzerox < (self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2] + margin)))
		right_lane_inds = ((nonzerox > (self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] - margin)) & 
			(nonzerox < (self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2] + margin)))
		# Again, extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds]
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]
		# Fit a second order polynomial to each
		self.left_fit = np.polyfit(lefty, leftx, 2)
		self.right_fit = np.polyfit(righty, rightx, 2)
		# Generate x and y values for plotting
		ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
		left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
		right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]
		return left_fitx, right_fitx

	def measure_curvature(self, raw, leftx, rightx):
		"""
		Compute the radius of curvature
		"""
		# Define conversions in x and y from pixels space to meters
		xm_per_pix = 3.7/raw.shape[1] # meters per pixel in x dimension
		ym_per_pix = 30/raw.shape[0] # meters per pixel in y dimension
		# Generate some fake data to represent lane-line pixels
		ploty = np.linspace(0, 719, num=720) # to cover same y-range as image
		# Fit second order polynomials to x, y in world space
		left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
		right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
		# Define y-value where we want radius of curvature
		# Choose the maximum y-value, corresponding to the bottom of the image
		y_eval = np.max(ploty)
		# Calculate radius of fitted curvature
		left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
		right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
		# Calculate the lane deviation
		lane_deviation = self.lane_deviation(raw, xm_per_pix)

		return left_curverad, right_curverad, lane_deviation

	def draw(self, binary, leftx, rightx):
		"""
		Project the measurement back down onto the road
		"""
		filled = np.zeros_like(binary)
		ploty = np.linspace(0, filled.shape[0] - 1, filled.shape[0])
		# Recast the x and y points into usable format for cv2.fillpoly()
		pts_left = np.array([np.transpose(np.vstack([leftx, ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([rightx, ploty])))])
		pts = np.hstack((pts_left, pts_right))
		# Draw the lane onto the warped blank image
		cv2.fillPoly(filled, np.int_([pts]), (0, 255, 0))
		return filled

	def calibrate_camera(self, calibration_images):
		"""
		Measuring Distortion
		"""
		# Chessboard calibration image corners
		nx = 9
		ny = 6
		# Arrays to store object points and image points from all the images
		objpoints = [] # 3D points in real world space
		imgpoints = [] # 2D points in image plane
		# Prepare object points
		objp = np.zeros((nx * ny, 3), np.float32)
		objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) # x, y coordinates
		print('Calibrating camera with chessboard images......')
		for fname in calibration_images:
			img = cv2.imread(fname)
			# Convert image to grayscale
			gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			# Find the chessboard corners
			ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
			# If found, store object points and corners
			if ret == True:
				objpoints.append(objp)
				imgpoints.append(corners)
		# Apply camera calibration function from CV
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (gray.shape[1], gray.shape[0]), None, None)
		# Create a dictionary to store the camera matrix and distortion coefficient
		camera_cal_data = {'matrix': mtx, 'distortion': dist}
		# Store calibrated data
		with open('camera_cal/camera_cal_data.p', 'wb') as f:
			pickle.dump(camera_cal_data, file=f)
			print('Finished calibrating camera, camera calibration data is stored in camera_cal/camera_cal_data.p')

	def cal_undistort(self, raw):
		"""
		Undistort raw images
		"""
		# Check the existence of camera calibration file
		# For executation efficiency, camera calibration should only need to do once
		if not os.path.exists('camera_cal/camera_cal_data.p'):
			# Read in and make a list of calibration images
			calibration_images = glob.glob('camera_cal/calibration*.jpg')
			self.calibrate_camera(calibration_images)	

		with open('camera_cal/camera_cal_data.p', 'rb') as f:
			camera_cal_data = pickle.load(file=f)

		return cv2.undistort(raw, camera_cal_data['matrix'], camera_cal_data['distortion'], None, camera_cal_data['matrix'])

	def perspective_transform(self, undistorted, direction='forward'):
		"""
		Convert image back and forth between birds'-eye view and camera view
		"""
		# Source image points
		src = np.float32([[255, 695], [585, 455], [700, 455], [1060, 690]])
		# Destination image points
		dst = np.float32([[305, 695], [305, 0], [1010, 0], [1010, 690]])
		# Perform forward or inverse perspective transform
		if direction == 'forward':
			# Compute the perspective transform, M
			M = cv2.getPerspectiveTransform(src, dst)
			# Create warped image - uses linear interpolation
			return cv2.warpPerspective(undistorted, M, (undistorted.shape[1], undistorted.shape[0]), flags=cv2.INTER_LINEAR)
		elif direction == 'inverse':
			# Compute the inverse also by swapping the input parameters
			Minv = cv2.getPerspectiveTransform(dst, src)
			return cv2.warpPerspective(undistorted, Minv, (undistorted.shape[1], undistorted.shape[0]), flags=cv2.INTER_LINEAR)

	def color_gradient(self, warped):

		# Apply color mask
		hsv = cv2.cvtColor(warped, cv2.COLOR_RGB2HSV)
		yellow_lower = np.array([20,60,60])
		yellow_upper = np.array([38,174,250])
		yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
		white_lower = np.array([202,202,202])
		white_upper = np.array([255,255,255])
		white_mask = cv2.inRange(warped, white_lower, white_upper)
		mask_binary = np.zeros_like(yellow_mask)
		mask_binary[(yellow_mask >= 1) | (white_mask >= 1)] = 1
		# Color and gradient threshold
		gray_thresh = (20, 255)
		s_thresh = (170, 255)
		l_thresh = (30, 255)
		# Convert image from RGB to HLS color space
		hls = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS)
		s_channel = hls[:, :, 2]
		l_channel = hls[:, :, 1]
		# Apply Sobel operator in x direction
		sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
		# Absolute x derivative to accentuate lines away from horizontal
		abs_sobelx = np.absolute(sobelx)
		scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
		# Generate a binary image
		sxbinary = np.zeros_like(scaled_sobelx)
		sxbinary[(scaled_sobelx >= gray_thresh[0]) & (scaled_sobelx <= gray_thresh[1])] = 1
		# Generate a binary image based on Saturation component of HLS color space
		s_binary = np.zeros_like(s_channel)
		s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
		# Generate a binary based on L component of HLS color space
		l_binary = np.zeros_like(l_channel)
		l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
		# Combining binary images
		binary = np.zeros_like(sxbinary)
		binary[( (l_binary == 1) & (s_binary == 1) & (mask_binary == 1) | (sxbinary == 1) )] = 1
		binary = 255 * np.dstack((binary, binary, binary)).astype('uint8')
		# Reduce the noise of binary image
		k = np.array([[1,1,1], [1,0,1], [1,1,1]])
		n = cv2.filter2D(binary, ddepth=-1, kernel=k)
		binary[n < 4] = 0
		return binary

	def lane_deviation(self, raw, xm_per_pix):
		# Calculate the intercept of fitted lane curvature at the bottom of image
		left_intercept = self.left_fit[0] * raw.shape[0] ** 2 + self.left_fit[1] * raw.shape[0] + self.left_fit[2]
		right_intercept = self.right_fit[0] * raw.shape[0] ** 2 + self.right_fit[1] * raw.shape[0] + self.right_fit[2]
		# Calculate lane deviation
		lane_center = (left_intercept + right_intercept) * 0.5
		return (lane_center - raw.shape[1] * 0.5) * xm_per_pix

	def process_pipeline(self, raw):
		# Calibrate camera and undistort images
		undistorted = self.cal_undistort(raw)
		# Apply perspective transform to bird's-eye view
		warped = self.perspective_transform(undistorted, direction='forward')
		# Apply color and gradient combined thresholding, and binarize
		binary = self.color_gradient(warped)
		# Smartly skip the sliding window to efficient find lanes
		if self.detected:
			left_fitx, right_fitx = self.lane_find(binary)
		else:
			left_fitx, right_fitx = self.basic_lane_find(binary)
		# Append containers of left and right lanes
		self.left_container[self.cursor] = left_fitx
		self.right_container[self.cursor] = right_fitx
		# Count the iteration
		self.cursor += 1
		if self.cursor >= self.average_n:
			self.cursor = 0
		# Averaging the lanes
		leftx = np.average(self.left_container, axis=0)
		rightx = np.average(self.right_container, axis=0)
		# Calculate the curvature
		left_curvature, right_curvature, lane_deviation = self.measure_curvature(raw, leftx, rightx)
		# Curvature info
		lane_cur_info = 'Left lane curvature: {:.2f} m; Right lane curvature: {:.2f} m'.format(left_curvature, right_curvature)
		# Lane deviation info
		lane_dev_info = 'Lane deviation: {:.2f} m'.format(lane_deviation)
		# Print road lane info text on raw image
		cv2.putText(undistorted, lane_cur_info, (90, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 50, 150), 3)
		cv2.putText(undistorted, lane_dev_info, (90, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 50, 150), 3)
		# Fill the lane
		filled = self.draw(binary, leftx, rightx)
		# Inversely perspective transform the image to the camera view, so could perform the image combination
		unwarped_processed_binary = self.perspective_transform(filled, direction='inverse')
		# Overlay processed images with raw image
		combined = cv2.addWeighted(undistorted, 1, unwarped_processed_binary, 0.3, 0)
		return combined