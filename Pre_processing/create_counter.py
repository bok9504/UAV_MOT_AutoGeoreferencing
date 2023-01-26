import cv2
import numpy as np
import yaml
import os
from Pre_processing.Enter_GeoCoordinates import Tk_GeoPoint

refPt = []
cap_img = []
count = 0

'''
# Counter를 구축하는 GUI 기반 프로그램 코드
 - 왼쪽 마우스 : 클릭한 좌표를 Counter point로 선정
 - 키보드 'r'  : 선택된 포인트를 차례로 삭제
 - 키보드 's'  : 선택된 포인트 저장
 - 키보드 'q'  : 저장하지 않고 GUI 프로그램 종료
'''
def create_counter_point(img_path, counter_point_file):

    print()
    print('The function selecting counter points ')
    print('- left mouse button : Select counter points')
    print('- keyboard "r" : remove selected points')
    print('- keyboard "s" : save selected points ')
    print('- keyboard "q" : exit without saving the selected points')
    print()

    def click_and_select(event, x, y, flags, param):
        global refPt, cap_img, count

        if event == cv2.EVENT_FLAG_LBUTTON:
            if len(refPt) == 0:
                refPt = [(x,y)]
            else:
                refPt.append((x,y))
            
            # counter point 선정 및 표출
            cv2.circle(image, (x, y), 9, (0, 0, 255), -1)
            if len(refPt) != 0 and len(refPt)%2 == 0 and count == 0:
                cv2.line(image, (refPt[0][0], refPt[0][1]), (refPt[1][0], refPt[1][1]), (0, 0, 255))
                count = count + 1
            elif len(refPt) != 0 and len(refPt)%2 == 0 and count == 1:
                cv2.line(image, (refPt[2][0], refPt[2][1]), (refPt[3][0], refPt[3][1]), (0, 0, 255))
                count = 0
            clone = image.copy()
            cap_img.append(clone)
            cv2.imshow('draw', cap_img[-1])

        elif event == cv2.EVENT_MOUSEWHEEL:pass

    image = cv2.imread(img_path)
    clone = image.copy()
    cap_img.append(clone)
    cv2.namedWindow('draw', cv2.WINDOW_NORMAL)
    cv2.imshow("draw", image)
    cv2.setMouseCallback("draw", click_and_select, cap_img[-1])
    cv2.waitKey(0)


    while True:
        cv2.imshow('draw', image)
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
            pointsFile = {'counter': {i: x for i, x in enumerate(refPt)}}
            with open(counter_point_file, 'w') as file:
                yaml.dump(pointsFile, file, sort_keys=False)
            break
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()

def get_counter_point(test_Video):
    counter_point_file = 'data/data_setting/counter_point/' + test_Video + '_point.yaml'
    first_frm = 'data/data_setting/source_img/' + test_Video + '/' + test_Video +'.jpg'
    
    if os.path.exists(counter_point_file):
        print()
        create_new_srcimg = input('If you wanna set new counter point, Write yes : ')
        if create_new_srcimg =='yes':
            create_counter_point(first_frm, counter_point_file)
        else:
            pass
    else:
        create_counter_point(first_frm, counter_point_file)