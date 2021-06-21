from scipy.optimize import least_squares
import numpy as np
import math
import cv2

# 삼변측량 수행 함수
def intersectionPoint(point_list, dist_list):

    pointX = []
    pointY = []
    
    for pointNum in range(len(point_list)):
        pointX.append(point_list[pointNum][0])
        pointY.append(point_list[pointNum][1])

    def eq(g):
        x, y = g
        
        pointResult = []
        for pointNum in range(len(pointX)):
            pointResult.append((x - pointX[pointNum])**2 + (y - pointY[pointNum])**2 - dist_list[pointNum]**2)
        pointResult = tuple(pointResult)
        return pointResult

    guess = (pointX[0], pointY[0] + dist_list[0])

    ans = least_squares(eq, guess, ftol=None, xtol=None)
    target_point = (round(ans.x[0]), round(ans.x[1]))

    return target_point

# 점1 x,y 와 점2 x,y의 거리 구하는 공식
def point_dist(R1,R2):
    
    a = R2[0] - R1[0]   # 선 a의 길이
    b = R2[1] - R1[1]    # 선 b의 길이
    
    dist = math.sqrt((a**2) + (b**2))    # (a * a) + (b * b)의 제곱근을 구함
    return dist

# Control Image Updating class
class update_ctlImg:
    def __init__(self, frm_point, center_point):
        self.frm_point = frm_point
        self.center_point = center_point

        self.datum_dist = []

    # frm_point와 Control Image의 center point 간의 거리 저장 (첫 프레임 기준)
    def get_datum_distance(self):
        for pointNum in range(len(self.frm_point)):
            for CPnum in range(len(self.center_point)):
                self.datum_dist.append(point_dist(self.center_point[CPnum], self.frm_point[pointNum]))
        self.datum_dist = np.reshape(self.datum_dist,(len(self.frm_point),len(self.center_point)))

    # 삼변측량을 활용한 frm_point 위치 갱신 (매 프레임 수행)
    def update_point(self, img, frm_point, centerPoint):
        for pointNum in range(len(frm_point)):
            img = cv2.circle(img, frm_point[pointNum], 10, (0,0,0),-1)
            newPoint = intersectionPoint(centerPoint, self.datum_dist[pointNum])
            self.frm_point[pointNum] = newPoint
        return self.frm_point