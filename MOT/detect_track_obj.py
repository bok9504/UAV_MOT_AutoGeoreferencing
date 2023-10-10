import numpy as np
import math
import cv2
from collections import Counter

from utilss import compute_color_for_labels, is_cross_pt
from utilss import PI, d2r, NormalizeAngle, image_angle, CoordConv, DBSCAN_clustering, Angle_append

class Obj_info:
    def __init__(self, bbox, cls, namess):
        self.bbox = bbox
        self.cls = cls
        self.label = []
        self.namess = namess
        self.geo_bbox = []

    # Draw Vehicle bounding boxes
    def draw_box(self, img, offset=(0,0)):
        for i, box in enumerate(self.bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            clsss = int(self.cls[i][0])
            color = compute_color_for_labels(clsss)
            t_size = cv2.getTextSize(self.label[i], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(
                img, (x1 + t_size[0] + 3, y1 - (t_size[1] + 4)), (x1, y1), color, -1)
            cv2.putText(img, self.label[i], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,255,255], 2)
        return img
    
    # GeoPoint Position calculation
    def calc_Geo_Position(self, geo_transform):
        for box in self.bbox:
            center = (int(((box[0]) + (box[2]))/2), int(((box[1]) + (box[3]))/2))
            geo_Cpoint = geo_transform * (center[1], center[0])
            self.geo_bbox.append(geo_Cpoint)
        return self.geo_bbox

class Detected_Obj(Obj_info):
    def __init__(self, bbox, cls, namess, confs):
        Obj_info.__init__(self, bbox, cls, namess)
        self.confs = confs

    # Setting bounding boxes label for each vehicle
    def set_label(self):
        for i in range(len(self.bbox)):
            clsss = int(self.cls[i][0])
            confss = float(self.confs[i][0])*100
            self.label.append('{} {:.2f}%'.format(self.namess[clsss], confss))

class Tracked_Obj(Obj_info):
    def __init__(self, bbox, cls, namess, id, pts, flag_drive, volume):
        Obj_info.__init__(self, bbox, cls, namess)
        self.id = id
        self.speed = []
        self.volume = volume
        self.heading = []
        self.heading_img = []
        self.track_heading = []
        self.track_heading_img = []
        self.flag_drive = flag_drive
        # Create the list of center points
        self.pts = pts
        for i, box in enumerate(self.bbox):
            center = (int(((box[0]) + (box[2]))/2), int(((box[1]) + (box[3]))/2))
            self.pts[self.id[i]].append(center)
        maxNum = max([len(self.pts[x]) for x in range(len(self.pts))])
        [self.pts[y].append(None) for y in range(len(self.pts)) if len(self.pts[y]) != maxNum]

    # Setting bounding boxes label for each vehicle
    def set_label(self):
        for i in range(len(self.bbox)):
            clsss = int(self.cls[i][0])
            ids = int(self.id[i])       
            if len(self.speed)==0 or self.speed[i] is None: # None speed information (just tracked)
                self.label.append('{}-{}'.format(self.namess[clsss], ids))
            else:   # speed information exists
                vehSpd = int(abs(self.speed[i]))
                self.label.append("{}-{} Speed:{}km/h".format(self.namess[clsss], ids, vehSpd))

    # Tracking path visualization
    def Visualize_Track(self, img):
        for i in range(len(self.id)):
            ptsTrk = self.pts[self.id[i]].copy()
            while None in ptsTrk:
                ptsTrk.remove(None)
            for j in range(1, len(ptsTrk)):
                cv2.line(img, (ptsTrk[j-1]), (ptsTrk[j]), (0,255,255), 5)
        return img

    # vehicle speed calculation
    def calc_Vehicle_Speed(self, FPS, dist_ratio, spd_interval): # set fps using spd_interval
        if FPS != 30: spd_term = 1
        else: spd_term = 3
        for i in range(len(self.id)):
            ptss = self.pts[self.id[i]].copy()
            curLoc = ptss.pop()
            ptss.reverse()
            if len(ptss) != 0:
                for frmIdx, prevLoc in enumerate(ptss):
                    if ((frmIdx+1)%(spd_term*spd_interval)==0) & (prevLoc != None):break    # Get previous vehicle location and frame index
                if frmIdx + 1 == len(ptss):   # Case of None previous vehicle location
                    self.speed.append(None)
                    continue
                frmMove_len = np.sqrt( pow(prevLoc[0] - curLoc[0], 2) + pow(prevLoc[1] - curLoc[1], 2) )    # Moving length in video frame
                geoMove_len = frmMove_len * dist_ratio      # Moving length in geo
                self.speed.append(geoMove_len * FPS * 3.6 / (frmIdx+1))
        return self.speed

    # vehicle speed calculation based on Georeferencing
    def geo_Vehicle_Speed(self, FPS, geo_transform, spd_interval): # set fps using spd_interval
        if FPS != 30: spd_term = 1
        else: spd_term = 3
        for i in range(len(self.id)):
            ptss = self.pts[self.id[i]].copy()
            curLoc = ptss.pop()
            ptss.reverse()
            if len(ptss) != 0:
                for frmIdx, prevLoc in enumerate(ptss):
                    if ((frmIdx+1)%(spd_term*spd_interval)==0) & (prevLoc != None):break    # Get previous vehicle location and frame index
                if frmIdx + 1 == len(ptss):   # Case of None previous vehicle location
                    self.speed.append(None)
                    continue
                geo_prevLoc = geo_transform * (prevLoc[1], prevLoc[0])
                geo_curLoc = geo_transform * (curLoc[1], curLoc[0])
                geoMove_len = np.sqrt( pow(geo_prevLoc[0] - geo_curLoc[0], 2) + pow(geo_prevLoc[1] - geo_curLoc[1], 2) )    # Moving length in video frame
                track_dir = image_angle(CoordConv(geo_curLoc[0], geo_curLoc[1]), CoordConv(geo_prevLoc[0], geo_prevLoc[1]))
                track_dir_img = image_angle((curLoc[0], curLoc[1]), (prevLoc[0], prevLoc[1]))                
                self.track_heading.append(track_dir)
                self.track_heading_img.append(track_dir_img)
                self.speed.append(geoMove_len * FPS * 3.6 / (frmIdx+1))
        return self.speed

    # traffic volume calculation
    def calc_Volume(self, Counter_list):
        for i in range(len(self.id)):
            ptss = self.pts[self.id[i]].copy()
            curLoc = ptss.pop()
            ptss.reverse()
            if len(ptss) != 0:
                for frmIdx, prevLoc in enumerate(ptss):
                    if (prevLoc != None) and (frmIdx%2==1):break
                if frmIdx + 1 == len(ptss): continue  # Case of None previous vehicle location
                for cntIdx, Counter in Counter_list.items():
                    if is_cross_pt(Counter[0][0], Counter[0][1], Counter[1][0], Counter[1][1], prevLoc[0], prevLoc[1], curLoc[0], curLoc[1]):
                        self.volume[cntIdx][self.cls[i]] += 1
                        self.volume[cntIdx][-1] += 1
        return self.volume
    
    # vehicle heading calculation
    def calc_Heading(self, img, geo_transform):
        '''
        헤딩 추정 코드
        1. 허프 선 변환 기반 각도 추정
         - 차량 객체 바운딩박스 내 허프 선 변환 통해 직선의 각도 검출
         - DBSCAN 클러스터링을 통해 각도값 군집 도출 이후, 최대 군집의 중앙값을 헤딩값 사용
        2. 주행방향 기반 헤딩값 보정
         - 주행 방향을 기반으로 반대 헤딩값 보정 수행
          1) 주행중 객체 (speed 8km/h 이상)
           - if 주행방향과 헤딩값 차이가 +- 45 이하: *=-1
          2) 정지 객체 (speed 8km/h 이하)
           - 주행기록x => pass, 주행기록o => 이전 헤딩값 사용
        '''
        for i, box in enumerate(self.bbox):
            box_img = img[box[1]:box[3], box[0]:box[2]]
            minLength = int(max(box_img.shape)/2)

            # 허프 선 변환 각도 검출
            edges = cv2.Canny(box_img, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, d2r, threshold=20, minLineLength=minLength, maxLineGap=10)
            angles = []
            angles_img = []
            if lines is not None:
                for line in lines:
                    _x1, _y1, _x2, _y2 = line[0]
                    geo_point1 = geo_transform * (_y1, _x1)
                    geo_point2 = geo_transform * (_y2, _x2)
                    angles.append(image_angle(CoordConv(geo_point1[1], geo_point1[0]), CoordConv(geo_point2[1], geo_point2[0])))
                    angles_img.append(image_angle((_x1, _y1), (_x2, _y2)))
                angles = np.array(angles)
                angles_img = np.array(angles_img)
                # 클러스터링을 통한 각도 필터링
                cluster_labels = DBSCAN_clustering(angles, epsilon = 2, min_samples = 3)
                angle_num = Counter(cluster_labels)
                if angle_num.most_common()[0][0] != -1:
                    angle_num_value = angle_num.most_common()[0][0]
                else:
                    if len(angle_num.most_common()) > 1: angle_num_value = angle_num.most_common()[1][0]
                    else: angle_num_value = angle_num.most_common()[0][0]
                angle_labels = cluster_labels == angle_num_value
                angles = angles[angle_labels]
                angles_img = angles_img[angle_labels]

                if (len(angles) % 2 == 0): angles = Angle_append(angles)
                if (len(angles_img) % 2 == 0): angles_img = Angle_append(angles_img)

                # 175 이상, -175 이하 각도 정규화 후 중앙값 계산 수행
                if (max(angles) >= 175 and min(angles) < -175): 
                    angles = np.vectorize(NormalizeAngle)(angles, False)
                    heading_angle = np.median(angles)
                else: heading_angle = NormalizeAngle(np.median(angles), False)
                if (max(angles_img) >= 175 and min(angles_img) < -175): 
                    angles_img = np.vectorize(NormalizeAngle)(angles_img, False)
                    heading_angle_img = np.median(angles_img)
                else: heading_angle_img = NormalizeAngle(np.median(angles_img), False)

                # heading_angle = np.median(angles)
                # heading_angle_img = np.median(angles_img)
            else: 
                heading_angle = 0
                heading_angle_img = 0
            
            # 반대 헤딩값 보정
            if len(self.speed) != 0:
                if self.speed[i] != None and self.speed[i] >= 8:
                    nor_angle = NormalizeAngle(self.track_heading[i], False) - NormalizeAngle(heading_angle, False)
                    nor_angle_img = NormalizeAngle(self.track_heading_img[i], False) - NormalizeAngle(heading_angle_img, False)
                    if nor_angle >= 135 or nor_angle <= -135:   # +-45도 기준
                        heading_angle = NormalizeAngle(heading_angle + 180, False)
                    if nor_angle_img >= 135 or nor_angle_img <= -135:   # +-45도 기준
                        heading_angle_img = NormalizeAngle(heading_angle_img + 180, False)
                    if len(self.flag_drive[self.id[i]]) == 0: 
                        self.flag_drive[self.id[i]].append(1)
                        self.flag_drive[self.id[i]].append(heading_angle)
                        self.flag_drive[self.id[i]].append(heading_angle_img)
                    else: 
                        self.flag_drive[self.id[i]].pop()
                        self.flag_drive[self.id[i]].pop()
                        self.flag_drive[self.id[i]].append(heading_angle)
                        self.flag_drive[self.id[i]].append(heading_angle_img)
                else:
                    if len(self.flag_drive[self.id[i]]) == 3 and self.flag_drive[self.id[i]][0] == 1: 
                        heading_angle = self.flag_drive[self.id[i]][1]
                        heading_angle_img = self.flag_drive[self.id[i]][2]
            self.heading.append(heading_angle)
            self.heading_img.append(heading_angle_img)
        return self.heading
    
    def heading_draw_box(self, img, offset=(0,0)):
        for i, box in enumerate(self.bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]

            radius = max(abs(x2 - x1), abs(y2 - y1))
            
            center_x = int((x1+x2)/2)
            center_y = int((y1+y2)/2)
            angle_rad = math.radians(self.heading_img[i])
            point_x = int(center_x + radius * math.cos(angle_rad))
            point_y = int(center_y + radius * math.sin(angle_rad))

            cv2.line(img, (center_x, center_y), (point_x, point_y), (0, 0, 255), 3)

            angle_rad = math.radians(self.heading[i])
            point_x = int(center_x + radius * math.cos(angle_rad))
            point_y = int(center_y + radius * math.sin(angle_rad))

            cv2.line(img, (center_x, center_y), (point_x, point_y), (0, 255, 0), 3)
        return img
