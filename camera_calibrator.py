import cv2
import numpy as np

class CameraCalibrator:
    def __init__(self):
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.blobDetector = CameraCalibrator._create_blod_detector()
        self.object_points = CameraCalibrator._create_objects_points()

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = None, None, None, None, None


    def calibrate(self, calibrate_image_files):

        objpoints = None  # 3d point in real world space
        imgpoints = None  # 2d points in image plane.
        for img_file in calibrate_image_files:
            img = cv2.imread(img_file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # keypoints = blobDetector.detect(gray)
            # im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            ret, corners = cv2.findCirclesGrid(gray, (4, 11),
                                               flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING,
                                               blobDetector=self.blobDetector)  # Find the circle grid
            # cv2_imshow(im_with_keypoints)
            # print(keypoints)
            if corners is not None and ret:
                objpoints = np.array([self.object_points]) if objpoints is None else np.append(objpoints, [self.object_points], axis=0)
                imgpoints = np.array([corners]) if imgpoints is None else np.append(imgpoints, [corners], axis=0)
            else:
                print(img_file)

        (i1, i2, i3, i4) = imgpoints.shape
        imgpoints = imgpoints.reshape([i1, i2, i4])
        print(imgpoints.shape)

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)




    def undistort(self, image_file):

        image = cv2.imread(image_file)

        (w, h) = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        dst = cv2.undistort(image, self.mtx, self.dist, None, newcameramtx)
        mapx,mapy = cv2.initUndistortRectifyMap(self.mtx, self.dist,None,newcameramtx,(w,h),5)
        # dst = cv2.remap(image,mapx,mapy,cv2.INTER_LINEAR)

        # x,y,w,h = roi
        # dst = dst[y:y+h, x:x+w]
        dst = cv2.resize(dst, (0, 0), fx=0.25, fy=0.25)

        return dst


    @staticmethod
    def _create_blod_detector():
        # Setup SimpleBlobDetector parameters.
        blobParams = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        blobParams.minThreshold = 30
        blobParams.maxThreshold = 255

        # Filter by Area.
        blobParams.filterByArea = True
        blobParams.minArea = 1364  # minArea may be adjusted to suit for your experiment
        blobParams.maxArea = 100000  # maxArea may be adjusted to suit for your experiment

        # Filter by Circularity
        blobParams.filterByCircularity = True
        blobParams.minCircularity = 0.4

        # Filter by Convexity
        blobParams.filterByConvexity = True
        blobParams.minConvexity = 0.87

        # Filter by Inertia
        blobParams.filterByInertia = True
        blobParams.minInertiaRatio = 0.01

        # Create a detector with the parameters
        blobDetector = cv2.SimpleBlobDetector_create(blobParams)

        return blobDetector


    @staticmethod
    def _create_objects_points():
        objp = np.zeros((44, 3), np.float32)
        for x in range(0, 11):
            for y in range(0, 4):
                objp[4 * x + y] = (x * 36, y * 72 + (x % 2) * 36, 0)


        return objp

