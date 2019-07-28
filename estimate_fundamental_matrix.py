import cv2 as cv
import numpy as np
from random import shuffle

def normalize2dpts(X):
    t = np.mean(X, 0)
    X0 = X - t
    s = np.sqrt(2) / np.linalg.norm(X0, axis=1).mean()
    T = np.array([[s, 0, -s * t[0]],
                  [0, s, -s * t[1]],
                  [0, 0,  1]])
    X0 *= s
    return X0, T


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



#there were used wiki:
#https://en.wikipedia.org/wiki/Eight-point_algorithm
# and material from lecture
    def find_fundamental_matrix(self):

        if self.points1 is None or self.points2 is None:
            raise Exception("There are no points yet")


        assert len(self.points1) == len(self.points2)

        size = len(self.points1)

        #let's build matrix Y, each is correspoiding to
        #[y1'*y1, y1'y2, y1', y2'y1, y2'y2, y2', y1, y2, 1]
        Y = np.zeros((size, 9))

        Y1, T1 = EstimateFMBy8Points._normalize2dpts(self.points1)
        Y2, T2 = EstimateFMBy8Points._normalize2dpts(self.points2)

        for i in range(size):
            y1, y2 = Y1[i][0], Y1[i][1]
            y1_hatch, y2_hatch = Y2[i][0], Y2[i][1]


            Y[i] = [y1_hatch*y1, y1_hatch*y2, y1_hatch,
                    y2_hatch*y1, y2_hatch*y2, y2_hatch,
                    y1_hatch, y2_hatch, 1]

        #SVD
        U, D, V = np.linalg.svd(Y)
        F = V[:, -1].reshape(3, 3)

        # force F to be rank  matrix
        U, D, V = np.linalg.svd(F, False)
        F = U.dot(np.diag((D[0], D[1], 0)).dot(V))

        # Denormalize
        F = T2.T.dot(F).dot(T1)

        self.F = F

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



#there were used code for this function from standard matlab normalize2dpts
#se here https://www.mathworks.com/matlabcentral/fileexchange/54544-normalise2dpts-pts?focused=5802540&tab=function&s_tid=mwa_osa_a
    @staticmethod
    def _normalize2dpts(pts):
        c = np.mean(pts, 0)
        newpts = pts - c
        dist = np.linalg.norm(newpts, axis=1)
        meandist = np.mean(dist)
        scale = np.sqrt(2) / meandist

        T = np.array([[scale, 0, -scale * c[0]],
                      [0, scale, -scale * c[1]],
                      [0, 0, 1]])
        newpts *= scale

        return newpts, T









