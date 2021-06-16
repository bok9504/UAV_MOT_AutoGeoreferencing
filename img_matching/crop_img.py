import argparse
import cv2
import os

refPt = []
cap_img = []

def create_control_img(img_path):

    print()
    print('The function creating control image ')
    print('- left mouse button : Create control image')
    print('- keyboard "r" : remove selected image ')
    print('- keyboard "s" : save selectde image ')
    print('- keyboard "q" : exit without saving the selectde image ')
    print()

    def click_and_crop(event, x, y, flags, param):    
        global refPt, cap_img
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(refPt) == 0:
                refPt = [(x,y)]
            else:
                refPt.append((x,y))

            cv2.rectangle(image, (refPt[-1][0]-256, refPt[-1][1]-256), (refPt[-1][0]+256, refPt[-1][1]+256), (0, 255, 0), 2)
            clone = image.copy()
            cap_img.append(clone)
            cv2.imshow('image', cap_img[-1])

    image = cv2.imread(img_path)
    crop_path = os.path.splitext(img_path)
    clone = image.copy()
    cap_img.append(clone)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', click_and_crop)
    cv2.moveWindow("image", 0,0)
    cv2.imshow("image", image)
    cv2.waitKey(0)

    while True:
        cv2.imshow('image', image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            if len(refPt) == 0:
                print("There's no selected image. Please create image using left mouse click.")
            elif len(refPt) == 1:
                del refPt[-1], cap_img[-1]
                image = clone.copy()
            else:
                del refPt[-1], cap_img[-1]
                image = cap_img[-1].copy()
        elif key == ord('s'):
            for imgNum in range(len(refPt)):
                roi = image[refPt[imgNum][1]-256:refPt[imgNum][1]+256, refPt[imgNum][0]-256:refPt[imgNum][0]+256]
                cv2.imwrite(crop_path[0] + '_ctl{}'.format(imgNum+1) + crop_path[1], roi)
            break
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()