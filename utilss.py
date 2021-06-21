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

