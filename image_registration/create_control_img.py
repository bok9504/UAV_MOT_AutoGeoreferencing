import cv2
import os

refPt = []
cap_img = []
'''
# Control Image를 생성하는 GUI 기반 프로그램 코드
 - 왼쪽 마우스 : 클릭한 좌표 중심으로 Control Image 생성
 - 키보드 'r'  : 선택된 이미지 차례로 삭제
 - 키보드 's'  : 선택된 이미지 저장
 - 키보드 'q'  : 저장하지 않고 GUI 프로그램 종료
'''
def create_control_img(img_path):

    print()
    print('The function creating control image ')
    print('- left mouse button : Create control image')
    print('- keyboard "r" : remove selected image ')
    print('- keyboard "s" : save selected image ')
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

# Control Image 존재여부 확인 및 생성코드
# - 존재하더라도 'yes' 입력 시 새로 생성가능
def get_control_img(test_Video):
    test_Video_folder = 'image_registration/control_img/' + test_Video
    test_Video_path = 'input_video/' + test_Video + '.MP4'
    first_frm = test_Video_folder + '/{}.jpg'.format(test_Video)
    if os.path.exists(test_Video_folder):
        print()
        create_new_ctlimg = input('If you wanna create new control image, Write yes : ')
        if create_new_ctlimg =='yes':
            create_control_img(first_frm)
        else:
            pass
    else:
        os.mkdir(test_Video_folder)
        vidcap = cv2.VideoCapture(test_Video_path)
        while(vidcap.isOpened()):
            ret, image = vidcap.read()
            cv2.imwrite(first_frm, image)
            break
        print(first_frm)
        create_control_img(first_frm)
    controlImg_path = [test_Video_folder + '/' + ctlImg for ctlImg in os.listdir(test_Video_folder) if 'ctl' in ctlImg]
    return controlImg_path