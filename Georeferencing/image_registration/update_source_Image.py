from scipy.optimize import least_squares
from functools import reduce
import numpy as np
import math

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

# Source Image Updating class
class update_srcImg:
    def __init__(self, center_point):
        self.center_point = center_point

    # Target point와 Source Image의 center point 간의 거리 저장 (첫 프레임 기준)
    def get_datum_distance(self, trg_point):
        _trg_point = [val for val in trg_point.values()]
        tmpshp = np.array(_trg_point).shape
        trg_list = np.array(_trg_point).flatten().tolist()
        datum_dist = []
        for pointNum in range(0, len(trg_list), 2):
            _trg_point_ = trg_list[pointNum:pointNum+2]
            for CPnum in range(len(self.center_point)):
                datum_dist.append(point_dist(self.center_point[CPnum], _trg_point_))
        datum_dist = np.reshape(datum_dist, (reduce(lambda x, y: x * y, tmpshp[:-1]), len(self.center_point)))
        return datum_dist

    # 삼변측량을 활용한 prev_point 위치 갱신 (매 프레임 수행)
    def update_point(self, trg_point, datum_dist):
        new_point = []
        _trg_point = [val for val in trg_point.values()]
        tmpshp = np.array(_trg_point).shape
        for pointNum in range(reduce(lambda x, y: x * y, tmpshp[:-1])):
            newPoint = intersectionPoint(self.center_point, datum_dist[pointNum])
            new_point.append(newPoint)
        new_point = np.reshape(new_point, tmpshp)
        new_point = dict(enumerate([tuple(x) for x in new_point]))
        return new_point