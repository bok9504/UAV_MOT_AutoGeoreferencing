import numpy as np
import cv2
import os
from pathlib import Path

def load_npz(path_npz):
    with np.load(path_npz) as X:
        ret, mtx, dist, _, _ = [X[i] for i in ('ret', 'mtx', 'dist', 'rvecs', 'tvecs')]
    return (ret, mtx, dist)

def get_undistort(img, cali_info):
    ret, mtx, dist = cali_info
    h,w,_ = img.shape
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    
    return dst

def writeVideo(filePath, cali_Video_path, path_npz):
    if os.path.isfile(filePath):
        cap = cv2.VideoCapture(filePath)
    else:
        print('파일이 존재하지 않습니다.')
    
    ret, image = cap.read()
    cali_info = load_npz(path_npz)
    frame = get_undistort(image, cali_info)
    
    width = frame.shape[1]
    height = frame.shape[0]
    fps = cap.get(cv2.CAP_PROP_FPS)
    fcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(cali_Video_path, fcc, fps, (width,height))
    
    while True:
        ret, frame = cap.read()
        
        if ret == False:break
        
        frame = get_undistort(frame, cali_info)
        out.write(frame)

    cap.release()
    out.release()

def check_caliVideo(test_Video, cali_npz):
    cali_Video_path = 'data/input_video/calibrated_video/' + test_Video + '.MP4'
    test_Video_path = 'data/input_video/' + test_Video + '.MP4'
    if os.path.exists(cali_Video_path) and\
         cv2.VideoCapture(cali_Video_path).get(cv2.CAP_PROP_FRAME_COUNT)+1 \
            == cv2.VideoCapture(test_Video_path).get(cv2.CAP_PROP_FRAME_COUNT):pass
    else:
        print('카메라 캘리브레이션 수행 중...')
        writeVideo(test_Video_path, cali_Video_path, cali_npz)
        print('카메라 캘리브레이션 완료')