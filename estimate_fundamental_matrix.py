import cv2 as cv
import numpy as np
from random import shuffle


class EstimateFMBy8Points:

    def __init__(self, image1, image2):
        self.surf = cv.xfeatures2d.SIFT_create(3000)
        self.brute_force_matcher = cv.BFMatcher_create()

        self.image1 = image1
        self.image2 = image2

        self.keypoints1 = None
        self.descrpitors1 = None

        self.keypoints2 = None
        self.descrpitors2 = None

        self.matches = None

        self.points1 = None
        self.points2 = None

        self.lines1 = None
        self.lines2 = None

        self.F = None

        self.COLOR = (0, 255, 0)

        self.SIZE = 8


    def run_find_matching(self, ):
        self.keypoints1, self.descrpitors1 = self.find_keypoints_and_descriptor(self.image1)
        self.keypoints2, self.descriptors2 = self.find_keypoints_and_descriptor(self.image2)

        matches = self.brute_force_matcher.match(self.descrpitors1, self.descriptors2)

        matches = list(sorted(matches, key = lambda x : x.distance))


        matches_20 = matches[:20]
        shuffle(matches_20)

        self.matches = matches_20[:8]

        self.points1, self.points2 = self.acquire_points_from_matches(self.matches, self.keypoints1, self.keypoints2)

        return  self.matches



    def find_fundamental_matrix(self):
        #TODO: add your implementation from scratch
        self.F, mask = cv.findFundamentalMat(self.points1, self.points2, method=cv.FM_8POINT)
        return self.F


    def find_epipolar_lines(self):

        if self.points1 is None or self.points2 is None:
            raise Exception("No mathcing points calculated yet")

        if self.F is None:
            raise Exception("Fun damental matrix is not calculated yet")

        self.lines1 = cv.computeCorrespondEpilines(self.points1, 1, self.F)
        self.lines1 = self.lines1.reshape(-1, 3)

        self.lines2 = cv.computeCorrespondEpilines(self.points2, 2, self.F)
        self.lines2 = self.lines2.reshape(-1, 3)


        return self.lines1, self.lines2

    def draw_matches(self):
        match_img = np.zeros((self.image1.shape[1] + self.image2.shape[1], self.image1.shape[0] + self.image2.shape[0]))
        match_img = cv.drawMatches(self.image1, self.keypoints1, self.image2, self.keypoints2,self.matches, match_img, flags=2)
        return match_img


    def draw_epilines_for_first_image(self):
        return EstimateFMBy8Points.draw_epilines(self.image1, self.lines1, self.points1, self.COLOR)

    def draw_epilines_for_second_image(self):
        return EstimateFMBy8Points.draw_epilines(self.image2, self.lines2, self.points2, self.COLOR)

    def find_keypoints_and_descriptor(self, image):
        keypoints, descritors = self.surf.detectAndCompute(image, None)

        return keypoints, descritors

    def acquire_points_from_matches(self, matches, keypoints1, keypoints2):
        points1 = []
        points2 = []
        for i in range(0, self.SIZE):
            p1_idx = matches[i].queryIdx
            points1.append(keypoints1[p1_idx].pt)

            p2_idx = matches[i].trainIdx
            points2.append(keypoints2[p2_idx].pt)

        points1 = np.array(points1)
        points2 = np.array(points2)

        return points1, points2


    @staticmethod
    def draw_epilines(img, lines, points, color=(0, 255, 0)):
        img_epilines = img.copy()

        size = len(lines)
        c = img_epilines.shape[1]
        for i in range(0, size):
            r = lines[i]
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

            cv.line(img_epilines, (x0, y0), (x1, y1), color, 1)
            p = (int(points[i][0]), int(points[i][1]))
            cv.circle(img_epilines, p, 5, color, -1)

        return img_epilines









