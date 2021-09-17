import numpy as np
import cv2
import time
'''
# Feature Matching(특징점 매칭)을 활용한 Image Registration(영상 정합) 구현 class
- 각각 descriptor, Matcher, matching_func 선정 후 인자값으로 전달
- descriptor : 'brisk', 'sift', 'orb'
- Matcher : 'bf', 'flann'
- Matching_func : 'match', 'knnMacth'
'''
class image_registration:
    def __init__(self, source_img, query_path, descriptor, feature_matcher, mathcing_func):
        self.source_img = source_img
        self.query_path = query_path

        self.descriptor = descriptor.lower()
        self.feature_matcher = feature_matcher.lower()
        self.mathcing_func = mathcing_func.lower()
        self.gray_img = []
        self.keypoints = []
        self.matcher = 0
        self.dst = []
        self.MIN_MATCH_COUNT = 5

    def gray_scale(self):
        img1 = cv2.imread(self.query_path)
        img2 = self.source_img
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        self.gray_img = [gray1, gray2]

    def description(self, thresh=30):
        if self.descriptor == 'brisk':
            detector = cv2.BRISK_create(thresh=thresh)
        elif self.descriptor == 'sift':
            detector = cv2.xfeatures2d.SIFT_create()
        elif self.descriptor == 'orb':
            detector = cv2.ORB_create()
        else:
            raise SystemExit('{} is invalid descriptor information.\n\
                Please write the correct descriptor.'.format(self.descriptor))

        kp1, des1 = detector.detectAndCompute(self.gray_img[0], None)
        kp2, des2 = detector.detectAndCompute(self.gray_img[1], None)
        
        self.keypoints= [(kp1, des1), (kp2, des2)]
        if (self.feature_matcher == 'flann') & (self.descriptor != 'orb'):
            self.keypoints= [(kp1, np.float32(des1)), (kp2, np.float32(des2))]


    def Matcher(self):
        if self.feature_matcher == 'bf':
            if self.descriptor == 'brisk' or self.descriptor == 'sift':
                normType = cv2.NORM_L1
            elif self.descriptor == 'orb':
                normType = cv2.NORM_HAMMING
            matcher = cv2.BFMatcher(normType)
        elif self.feature_matcher == 'flann':
            if self.descriptor == 'brisk' or self.descriptor == 'sift':
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
            elif self.descriptor == 'orb':
                FLANN_INDEX_LSH = 6
                index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6, 
                                key_size = 12, multi_probe_level = 1)
                search_params = dict(checks=32)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise SystemExit('{} is invalid matcher information.\n\
                Please write the correct matcher.'.format(self.feature_matcher))
        self.matcher = matcher

    def feature_matching(self, t_acc_check = False):
        if self.mathcing_func == 'match':
            matches = self.matcher.match(self.keypoints[0][1], self.keypoints[1][1])
            matches = sorted(matches, key=lambda x:x.distance)
            min_dist, max_dist = matches[0].distance, matches[-1].distance
            ratio = 0.2
            good_thresh = (max_dist - min_dist) * ratio + min_dist
            good_matches = [m for m in matches if m.distance < good_thresh]        

        elif self.mathcing_func == 'knnmatch':
            matches = self.matcher.knnMatch(self.keypoints[0][1], self.keypoints[1][1], k=2)
            ratio = 0.75
            good_matches = [first for first,second in matches if first.distance < second.distance * ratio]

        elif self.mathcing_func == 'radiusmatch':
            raise SystemExit('Disable radiusMatch. Please use Match or knnMatch.')
        else:
            raise SystemExit('{} is invalid match information.\n\
                Please write the correct match.'.format(self.mathcing_func))
        if len(good_matches)>=self.MIN_MATCH_COUNT:
            if t_acc_check: print('acc : {:0.2f}%'.format(len(good_matches)/len(matches)*100))
            src_pts = np.float32([self.keypoints[0][0][m.queryIdx].pt for m in good_matches ])
            dst_pts = np.float32([self.keypoints[1][0][m.trainIdx].pt for m in good_matches ])
            mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h,w = self.gray_img[0].shape[:2]
            pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
            dst = cv2.perspectiveTransform(pts,mtrx)  
            self.dst = dst
        else:

            raise SystemExit("Not enough matches are found - {}/{}\n{}".format(len(good_matches), self.MIN_MATCH_COUNT, self.query_path))

    def draw_Line_get_ConterPoint(self):
        dst = np.int32(self.dst)
        self.source_img = cv2.polylines(self.source_img, [dst], True, 255,3, cv2.LINE_AA)
        center_point = (int((dst[0][0][0] + dst[2][0][0])/2), int((dst[0][0][1] + dst[2][0][1])/2))
        return center_point

def run_image_registration(source_img, query_path, descriptor, feature_matcher, mathcing_func, t_acc_check = False):    
    start = time.time()
    imgMatching = image_registration(source_img, query_path, descriptor, feature_matcher, mathcing_func)
    imgMatching.gray_scale()
    imgMatching.description(60)
    imgMatching.Matcher()
    imgMatching.feature_matching(t_acc_check)
    center_point = imgMatching.draw_Line_get_ConterPoint()

    if t_acc_check:print('time: {:0.4f}'.format(time.time() - start))
    return center_point