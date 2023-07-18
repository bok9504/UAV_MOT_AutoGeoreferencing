import sys
sys.path.insert(0, './MOT/yolov5')

from MOT.yolov5.models.common import DetectMultiBackend
from MOT.yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from MOT.yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                                      increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from MOT.yolov5.utils.torch_utils import select_device, time_sync, smart_inference_mode
from MOT.yolov5.utils.plots import Annotator, colors, save_one_box
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
import platform
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
    project, source, weights, view_img, save_txt, imgsz, src_img, \
        nosave, name, exist_ok, dnn, half, vid_stride, augment, classes, agnostic_nms, max_det, save_crop, line_thickness, save_conf, hide_labels, hide_conf, update, device, data, visualize, save_label = \
        opt.project, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.src_img, \
            opt.nosave, opt.name, opt.exist_ok, opt.dnn, opt.half, opt.vid_stride, opt.augment, opt.classes, opt.agnostic_nms, opt.max_det, opt.save_crop, opt.line_thickness, opt.save_conf, opt.hide_labels, opt.hide_conf, opt.update, opt.device, opt.data, opt.visualize, opt.save_label
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

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_label else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    txt_path = str(Path(save_dir)) + '/results.txt'

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    global namess
    namess = model.module.names if hasattr(model, 'module') else model.names

    # Create the list of center points using deque
    from _collections import deque
    pts = [deque(maxlen=200) for _ in range(10000)]

    # Get control point & counter point
    if img_registration_swch:
        with open('data/data_setting/control_point/'+test_Video+'_point.yaml') as f:
            data_CtP = yaml.load(f.read()) 
        frm_point = data_CtP['frm_point']
        geo_point = data_CtP['geo_point']
        if Georeferencing_swch and img_registration_swch:
            gcps = [GCP(frm_point[x][0], frm_point[x][1], geo_point[x][0], geo_point[x][1]) for x in range(len(frm_point))]
            geo_transform = from_gcps(gcps)
        if volume_swch:
            with open('data/data_setting/counter_point/'+test_Video+'_point.yaml') as f:
                data_CoP = yaml.load(f.read())
            Counter_list = {0 : [data_CoP['counter'][0], data_CoP['counter'][1]], 1 : [data_CoP['counter'][2], data_CoP['counter'][3]]}
            volume = np.zeros((len(Counter_list),len(namess)+1))
    
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process detections
        for det_idx, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[det_idx], im0s[det_idx].copy(), dataset.count
                s += f'{det_idx}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            label_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

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
                    for counter in Counter_list.values():
                        im0 = cv2.line(im0, tuple(counter[0]), tuple(counter[1]), (0,0,0), 5,-1)
                    for cntIdx in range(len(Counter_list)):
                        cv2.putText(im0, 'count_{}_total : {}'.format(cntIdx+1, volume[cntIdx][-1]), (100+400*cntIdx, 110), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2) # 카운팅 되는거 보이게
                        cv2.putText(im0, 'count_{}_{} : {}'.format(cntIdx+1, namess[0], volume[cntIdx][0]), (100+400*cntIdx, 140), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2) # 카운팅 되는거 보이게
                        cv2.putText(im0, 'count_{}_{} : {}'.format(cntIdx+1, namess[1], volume[cntIdx][1]), (100+400*cntIdx, 170), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2) # 카운팅 되는거 보이게
                        cv2.putText(im0, 'count_{}_{} : {}'.format(cntIdx+1, namess[2], volume[cntIdx][2]), (100+400*cntIdx, 200), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2) # 카운팅 되는거 보이게

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                bbox_xywh = []
                bbox_xyxy = []
                confs = []
                clss = []

                for *xyxy, conf, cls in reversed(det):
                    # Adapt detections to deep sort input format
                    x_c, y_c, bbox_w, bbox_h = bbox_ccwh(*xyxy)
                    x_l, y_t, x_r, y_d = bbox_ltrd(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    objxyxy = [x_l, y_t, x_r, y_d]
                    bbox_xywh.append(obj)
                    bbox_xyxy.append(objxyxy)
                    confs.append([conf.item()])
                    clss.append([cls.item()])
                    
                    # Write results
                    if save_label:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{label_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if not hide_labels:
                        if save_img or save_crop:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = names[c] if hide_conf else f'{names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # draw detected boxes for visualization
                if yolo_swch:
                    detect_result = Detected_Obj(bbox_xyxy, clss, namess, confs)
                    detect_result.set_label()
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
                    track_result = Tracked_Obj(bbox_xyxy, cls_id, namess, identities, pts, volume if volume_swch else None)
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
                            volume = track_result.calc_Volume(Counter_list)
                    track_result.set_label()
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

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[det_idx] != save_path:  # new video
                        vid_path[det_idx] = save_path
                        if isinstance(vid_writer[det_idx], cv2.VideoWriter):
                            vid_writer[det_idx].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[det_idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[det_idx].write(im0)
        
        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_label or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_label else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

if __name__ == '__main__':

    # Choose Function (True/False)
    camera_calibrate_switch = False  # 카메라 캘리브레이션
    yolo_switch = False              # 차량 객체 검지 표출
    deepsort_switch = True         # 차량 객체 추적 표출
    VehTrack_switch = False         # 차량 주행궤적 추출
    img_registration_switch = True # 영상 정합 수행
    speed_switch = False            # 차량별 속도 추출 (영상정합 필요)
    volume_switch = True           # 교통량 추출      (영상정합 필요)
    Georeferencing_switch = False   # 지오레퍼런싱 (영상정합 필요)

    # Setting Parameters
    test_Video = 'sample_video' # 테스트 영상 이름
    exp_num = 'test' # 실험 이름

    weights_path = 'MOT/yolov5/weights/yolov5s.pt'
    test_Video_path = 'data/input_video/' + test_Video + '.MP4'  # 테스트할 영상 경로 입력
    output_path = 'data/output_folder/' + test_Video + '_' + exp_num  # 실험결과 저장 경로
    cali_npz = 'data/data_setting/calibration_info/mavic2_pro.npz'       # 카메라 캘리브레이션 정보

    img_size = 640 # 이미지 사이즈(default : 640) : 이미지의 크기를 조절(resizing)하여 검출하도록 만듦, 크면 클수록 검지율이 좋아지지만 FPS가 낮아짐
    conf_thres = 0.4  # 신뢰도 문턱값(default : 0.4) : 해당 수치 검지율 이하는 제거, Yolov5 학습결과(F1_curve.png) 보고 설정 But. 보통 경험적으로 설정
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
    parser.add_argument('--weights', type=str, default=weights_path, help='model.pt path')
    parser.add_argument('--source', type=str, default=test_Video_path, help='source')   # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default=output_path, help='output folder')  # output folder
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[img_size, img_size], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=conf_thres, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=iou_thres, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default=device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', default=True, help='save results to *.txt')
    parser.add_argument('--save-label', action='store_true', help='save label results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=classes_type, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=output_path, help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument("--config_deepsort", type=str, default="MOT/deep_sort_pytorch/configs/deep_sort.yaml")
    
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

