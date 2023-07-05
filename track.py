import sys
sys.path.insert(0, './MOT/yolov5')

from MOT.yolov5.models.experimental import attempt_load
from MOT.yolov5.utils.datasets import LoadImages, LoadStreams
from MOT.yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow
from MOT.yolov5.utils.torch_utils import select_device, time_synchronized
from MOT.deep_sort_pytorch.utils.parser import get_config
from MOT.deep_sort_pytorch.deep_sort import DeepSort
from MOT.detect_track_obj import Detected_Obj
from MOT.detect_track_obj import Tracked_Obj

from Pre_processing.camera_calibration import check_caliVideo
from Pre_processing.create_source_img import get_source_img
from Pre_processing.create_control_point import get_control_point
from Pre_processing.create_counter import get_counter_point

from Georeferencing.image_registration.feature_matching import run_image_registration
from Georeferencing.image_registration.update_source_Image import update_srcImg, point_dist

from utilss import bbox_ccwh
from utilss import bbox_cc
from utilss import bbox_ltrd

import argparse
import os
import platform
import shutil
import time
import yaml
import numpy as np
from scipy import stats
from pathlib import Path
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP
import cv2
import torch
import torch.backends.cudnn as cudnn

def detect(opt):
    out, source, weights, view_vid, save_vid, save_txt, imgsz, src_img = \
        opt.output, opt.source, opt.weights, opt.view_vid, opt.save_vid, opt.save_txt, opt.img_size, opt.src_img
    yolo_swch, deepsort_swch, img_registration_swch, vehtrk_swch, speed_swch, volume_swch, Georeferencing_swch = \
        opt.yolo_swch, opt.deepsort_swch, opt.img_registration_swch, opt.vehtrk_swch, opt.speed_swch, opt.volume_swch, opt.Georeferencing_swch
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None 
    # Check if environment supports image displays
    if view_vid:
        view_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    global namess
    namess = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    # txt file path
    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    # Create the list of center points using deque
    from _collections import deque
    pts = [deque(maxlen=200) for _ in range(10000)]

    # Get control point & counter point
    if img_registration_swch:
        with open('data/data_setting/control_point/'+test_Video+'_point.yaml') as f:
            data = yaml.load(f.read()) 
        frm_point = data['frm_point']
        geo_point = data['geo_point']
        if Georeferencing_swch and img_registration_swch:
            gcps = [GCP(frm_point[x][0], frm_point[x][1], geo_point[x][0], geo_point[x][1]) for x in range(len(frm_point))]
            geo_transform = from_gcps(gcps)
        if volume_swch:
            with open('data/data_setting/counter_point/'+test_Video+'_point.yaml') as f:
                data = yaml.load(f.read())
            Counter_list = {0 : [data['counter'][0], data['counter'][1]], 1 : [data['counter'][2], data['counter'][3]]}
            volume = np.zeros((len(Counter_list),len(namess)+1))

    # Start loop
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if speed_swch:
                print()
                dist_ratio_list = []
                for i in range(len(frm_point)):
                    for j in range(i+1, len(geo_point)):
                        dist_ratio_list.append(point_dist(geo_point[i],geo_point[j])/point_dist(frm_point[i],frm_point[j]))
                dist_ratio = stats.trim_mean(dist_ratio_list, 0.5)

            # Frame Points update using image registration
            if img_registration_swch:
                # image registration
                centerPoint = []
                for src_img_path in src_img:
                    centerPoint.append(run_image_registration(im0, src_img_path, 'brisk', 'bf', 'knnmatch'))
                srcImg_centerPoint = update_srcImg(centerPoint)
                # Updating Frame Points
                if frame_idx==0:
                    datum_dist_frm = srcImg_centerPoint.get_datum_distance(frm_point)
                frm_point = srcImg_centerPoint.update_point(frm_point, datum_dist_frm)
                for pointNum in range(len(frm_point)):
                    im0 = cv2.circle(im0, frm_point[pointNum], 10, (0,0,0),-1)
                # Updating Counters
                if volume_swch:
                    if frame_idx==0:
                        datum_dist_cnt = srcImg_centerPoint.get_datum_distance(Counter_list)
                    Counter_list = srcImg_centerPoint.update_point(Counter_list, datum_dist_cnt)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, namess[int(c)])  # add to string

                bbox_xywh = []
                bbox_xyxy = []
                confs = []
                clss = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_ccwh(*xyxy)
                    x_l, y_t, x_r, y_d = bbox_ltrd(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    objxyxy = [x_l, y_t, x_r, y_d]
                    bbox_xywh.append(obj)
                    bbox_xyxy.append(objxyxy)
                    confs.append([conf.item()])
                    clss.append([cls.item()])

                # draw detected boxes for visualization
                if yolo_swch:
                    detect_result = Detected_Obj(bbox_xyxy, clss, confs)
                    detect_result.set_label(namess)
                    detect_result.draw_box(im0)

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                cls_ids = torch.Tensor(clss)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, cls_ids, im0)
                """
                # outputs shape

                [[박스 좌측상단 x, 박스 좌측상단 y, 박스 우측하단 x, 박스 우측하단 y, 클래스 넘버, 차량 id],
                [박스 좌측상단 x, 박스 좌측상단 y, 박스 우측하단 x, 박스 우측하단 y, 클래스 넘버, 차량 id],
                [박스 좌측상단 x, 박스 좌측상단 y, 박스 우측하단 x, 박스 우측하단 y, 클래스 넘버, 차량 id],
                [박스 좌측상단 x, 박스 좌측상단 y, 박스 우측하단 x, 박스 우측하단 y, 클래스 넘버, 차량 id],
                ...]
                """
                # Calculates the Geo point for each vehicle
                if Georeferencing_swch and img_registration_swch:
                    geo_bbox = []
                    for output in outputs:
                        x_c, y_c = bbox_cc(output[0:4])
                        geo_Cpoint = geo_transform * (y_c, x_c)
                        geo_bbox.append(geo_Cpoint)

                # draw tracked boxes for visualization
                if deepsort_swch and len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    cls_id = outputs[:,4:5]
                    identities = outputs[:, -1]
                    track_result = Tracked_Obj(bbox_xyxy, cls_id, identities, pts)
                    # draw vehicle trajectory for visualization
                    if vehtrk_swch:
                        track_result.Visualize_Track(im0)
                    # calculate vehicle speed
                    if speed_swch and img_registration_swch and not Georeferencing_swch:
                        veh_speed = track_result.calc_Vehicle_Speed(vid_cap, dist_ratio, 1)
                    elif speed_swch and img_registration_swch and Georeferencing_swch:
                        veh_speed = track_result.geo_Vehicle_Speed(vid_cap, geo_transform, 1)
                    # calculate vehicle volume
                    if volume_swch and img_registration_swch:
                        if frame_idx % 2 == 0:
                            volume = track_result.calc_Volume(Counter_list, volume)
                        track_result.draw_Volume(im0, Counter_list, volume, namess)
                    track_result.set_label(namess)
                    track_result.draw_box(im0)

                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_right = output[2]
                        bbox_down = output[3]
                        cls_id = output[4]
                        identity = output[-1]
                        if speed_swch and img_registration_swch and len(veh_speed)!=0 and veh_speed[j]!=None \
                            and img_registration_swch: spd = veh_speed[j]
                        else: spd = -1
                        if Georeferencing_swch and img_registration_swch:
                            geo_x = geo_bbox[j][0]
                            geo_y = geo_bbox[j][1]
                        else: geo_x = geo_y = -1
                            
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 8 + '%f ' * 2 + '\n') % (frame_idx, identity, bbox_left,
                                                           bbox_top, bbox_right, bbox_down, cls_id, spd, geo_x, geo_y))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':

    # Choose Function (True/False)
    camera_calibrate_switch = False  # 카메라 캘리브레이션
    yolo_switch = False              # 차량 객체 검지 표출
    deepsort_switch = True         # 차량 객체 추적 표출
    VehTrack_switch = False         # 차량 주행궤적 추출
    img_registration_switch = True # 영상 정합 수행
    speed_switch = True            # 차량별 속도 추출 (영상정합 필요)
    volume_switch = True           # 교통량 추출      (영상정합 필요)
    Georeferencing_switch = True   # 지오레퍼런싱 (영상정합 필요)

    # Setting Parameters
    test_Video = 'sample_video' # 테스트 영상 이름
    exp_num = 'test' # 실험 이름

    weights_path = 'MOT/yolov5/weights/yolo_weight.pt'
    test_Video_path = 'data/input_video/' + test_Video + '.MP4'  # 테스트할 영상 경로 입력
    output_path = 'data/output_folder/' + test_Video + '_' + exp_num  # 실험결과 저장 경로
    cali_npz = 'data/data_setting/calibration_info/mavic2_pro.npz'       # 카메라 캘리브레이션 정보

    img_size = 800 # 이미지 사이즈(default : 640) : 이미지의 크기를 조절(resizing)하여 검출하도록 만듦, 크면 클수록 검지율이 좋아지지만 FPS가 낮아짐
    conf_thres = 0.478  # 신뢰도 문턱값(default : 0.4) : 해당 수치 검지율 이하는 제거, Yolov5 학습결과(F1_curve.png) 보고 설정 But. 보통 경험적으로 설정
    iou_thres = 0.1  # iou 문턱값(default : 0.5) : 검출 박스의 iou(교집합) 정도
    classes_type = [0, 1, 2] # 데이터셋 및 학습된 모델 클래스 종류
    device = 'cpu'


    # 카메라 캘리브레이션 수행
    if camera_calibrate_switch:
        check_caliVideo(test_Video, cali_npz)
        test_Video = test_Video + '_cali'
        test_Video_path = 'data/input_video/' + test_Video + '.MP4'

    # Source Image, control_point, counter_point 존재 여부 확인 후, 없으면 생성, 있으면 데이터 로드
    if img_registration_switch:
        Source_Img_path = get_source_img(test_Video)
        get_control_point(test_Video)
        get_counter_point(test_Video)
    else:Source_Img_path=[]

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default=weights_path, help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default=test_Video_path, help='source')
    parser.add_argument('--output', type=str, default=output_path,
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=img_size,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=conf_thres, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=iou_thres, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default=device,
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view_vid', action='store_false', default=True,
                        help='display results')
    parser.add_argument('--save-vid', action='store_true', default=output_path,
                        help='display results')
    parser.add_argument('--save-txt', action='store_true', default=True,
                        help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int,
                        default=classes_type, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', default=True,
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="MOT/deep_sort_pytorch/configs/deep_sort.yaml")
    
    parser.add_argument("--src_img", default=Source_Img_path, help='Source Images path')
    parser.add_argument("--yolo_swch", default=yolo_switch, help='Yolo on & off')
    parser.add_argument("--deepsort_swch", default=deepsort_switch, help='DeepSORT on & off')
    parser.add_argument("--img_registration_swch", default=img_registration_switch, help='Image Registration on & off')
    parser.add_argument("--vehtrk_swch", default=VehTrack_switch, help='Vehicle Track on & off')
    parser.add_argument("--speed_swch", default=speed_switch, help='Speed on & off')
    parser.add_argument("--volume_swch", default=volume_switch, help='Volume on & off')
    parser.add_argument("--Georeferencing_swch", default=Georeferencing_switch, help='Georeferencing on & off')
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)

