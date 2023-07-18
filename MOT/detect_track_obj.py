import numpy as np
import cv2

from utilss import compute_color_for_labels
from utilss import is_cross_pt

class Obj_info:
    def __init__(self, bbox, cls, namess):
        self.bbox = bbox
        self.cls = cls
        self.label = []
        self.namess = namess

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
    def __init__(self, bbox, cls, namess, id, pts, volume):
        Obj_info.__init__(self, bbox, cls, namess)
        self.id = id
        self.speed = []
        self.volume = volume
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
    def calc_Vehicle_Speed(self, vid_cap, dist_ratio, spd_interval): # set fps using spd_interval
        for i in range(len(self.id)):
            ptss = self.pts[self.id[i]].copy()
            curLoc = ptss.pop()
            ptss.reverse()
            if len(ptss) != 0:
                for frmIdx, prevLoc in enumerate(ptss):
                    if ((frmIdx+1)%(3*spd_interval)==0) & (prevLoc != None):break    # Get previous vehicle location and frame index
                if frmIdx + 1 == len(ptss):   # Case of None previous vehicle location
                    self.speed.append(None)
                    continue
                frmMove_len = np.sqrt( pow(prevLoc[0] - curLoc[0], 2) + pow(prevLoc[1] - curLoc[1], 2) )    # Moving length in video frame
                geoMove_len = frmMove_len * dist_ratio      # Moving length in geo
                self.speed.append(geoMove_len * vid_cap.get(cv2.CAP_PROP_FPS) * 3.6 / (frmIdx+1))
        return self.speed

    # vehicle speed calculation based on Georeferencing
    def geo_Vehicle_Speed(self, vid_cap, geo_transform, spd_interval): # set fps using spd_interval
        for i in range(len(self.id)):
            ptss = self.pts[self.id[i]].copy()
            curLoc = ptss.pop()
            ptss.reverse()
            if len(ptss) != 0:
                for frmIdx, prevLoc in enumerate(ptss):
                    if ((frmIdx+1)%(3*spd_interval)==0) & (prevLoc != None):break    # Get previous vehicle location and frame index
                if frmIdx + 1 == len(ptss):   # Case of None previous vehicle location
                    self.speed.append(None)
                    continue
                geo_prevLoc = geo_transform * (prevLoc[1], prevLoc[0])
                geo_curLoc = geo_transform * (curLoc[1], curLoc[0])
                geoMove_len = np.sqrt( pow(geo_prevLoc[0] - geo_curLoc[0], 2) + pow(geo_prevLoc[1] - geo_curLoc[1], 2) )    # Moving length in video frame
                self.speed.append(geoMove_len * vid_cap.get(cv2.CAP_PROP_FPS) * 3.6 / (frmIdx+1))
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