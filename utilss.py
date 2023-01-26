import time
import numpy as np

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    label = (label + 1) * 2
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def bbox_ccwh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def bbox_ltrd(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_right = max([xyxy[0].item(), xyxy[2].item()])
    bbox_down = max([xyxy[1].item(), xyxy[3].item()])
    return bbox_left, bbox_top, bbox_right, bbox_down

def bbox_cc(output):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([output[0], output[2]])
    bbox_top = min([output[1], output[3]])
    bbox_w = abs(output[0] - output[2])
    bbox_h = abs(output[1] - output[3])
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    return x_c, y_c

def track_time(func):
    def new_func(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        exec_time = end_time - start_time
        print('"{}" function : {:.4f} sec'.format(func.__name__, exec_time))
    return new_func

def is_divide_pt(x11,y11, x12,y12, x21,y21, x22,y22):
    f1= (x12-x11)*(y21-y11) - (y12-y11)*(x21-x11)
    f2= (x12-x11)*(y22-y11) - (y12-y11)*(x22-x11)
    if np.sign(f1)*np.sign(f2) < 0 :
        return True
    else:
        return False

def is_cross_pt(x11,y11, x12,y12, x21,y21, x22,y22):
    b1 = is_divide_pt(x11,y11, x12,y12, x21,y21, x22,y22)
    b2 = is_divide_pt(x21,y21, x22,y22, x11,y11, x12,y12)
    if b1 and b2:
        return True
    else:
        return False

# import sys
# sys.path.append("..")
# from utilss import track_time