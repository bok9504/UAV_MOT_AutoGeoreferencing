import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, \
    check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from img_matching.crop_img import create_control_img

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn



palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


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

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    label = (label + 1) * 2
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

class Obj_info:
    def __init__(self, bbox, cls):
        self.bbox = bbox
        self.cls = cls

class Detected_Obj(Obj_info):
    def __init__(self, bbox, cls, confs):
        Obj_info.__init__(self, bbox, cls)
        self.confs = confs

    def draw_box(self, img, offset=(0,0)):
        for i, box in enumerate(self.bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            clsss = int(self.cls[i][0])
            confss = float(self.confs[i][0])*100
            color = compute_color_for_labels(clsss)
            label = '{} {:.2f}%'.format(namess[clsss], confss)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(
                img, (x1 + t_size[0] + 3, y1 - (t_size[1] + 4)), (x1, y1), color, -1)
            cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,255,255], 2)
        return img

class Tracked_Obj(Obj_info):
    def __init__(self, bbox, cls, id):
        Obj_info.__init__(self, bbox, cls)
        self.id = id

    def draw_box(self, img, offset=(0,0)):
        for i, box in enumerate(self.bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            clsss = int(self.cls[i][0])
            ids = int(self.id[i])
            color = compute_color_for_labels(clsss)
            label = '{}-{}'.format(namess[clsss], ids)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(
                img, (x1 + t_size[0] + 3, y1 - (t_size[1] + 4)), (x1, y1), color, -1)
            cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,255,255], 2)
        return img

    def Visualize_Track(self, img, pts):
        for i, box in enumerate(self.bbox):
            center = (int(((box[0]) + (box[2]))/2), int(((box[1]) + (box[3]))/2))
            pts[self.id[i]].append(center)
            for j in range(1, len(pts[self.id[i]])):
                if pts[self.id[i]][j-1] is None or pts[self.id[i]][j] is None:
                    continue
                cv2.line(img, (pts[self.id[i]][j-1]), (pts[self.id[i]][j]), (0,255,255), 5)
        return img

def detect(opt):
    out, source, weights, view_vid, save_vid, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_vid, opt.save_vid, opt.save_txt, opt.img_size
    yolo_swch, deepsort_swch, vehtrk_swch, speed_swch, volume_swch, line_swch, density_swch, headway_swch = \
        opt.yolo_swch, opt.deepsort_swch, opt.vehtrk_swch, opt.speed_swch, opt.volume_swch, opt.line_swch, opt.density_swch, opt.headway_swch
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

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    from _collections import deque
    pts = [deque(maxlen=100) for _ in range(10000)]

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

                # draw tracked boxes for visualization
                if deepsort_swch and len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    cls_id = outputs[:,4:5]
                    identities = outputs[:, -1]
                    track_result = Tracked_Obj(bbox_xyxy, cls_id, identities)
                    track_result.draw_box(im0)

                    # draw vehicle trajectory for visualization
                    if vehtrk_swch:
                        track_result.Visualize_Track(im0, pts)

                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

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

    # YoloV5 + DeepSORT 트래킹 수행

    # 표출 기능 선택
    yolo_switch = False             # 차량 객체 검지 표출
    deepsort_switch = True          # 차량 객체 추적 표출
                                    # - yolo와 deepsort는 둘중 하나만 True 선택 (중복선택 시 중복된 결과물 표출)
    VehTrack_switch = True          # 차량 주행궤적 추출
    speed_switch = True            # 차량별 속도 추출
    volume_switch = False           # 교통량 추출
    line_switch = False             # 차선 추출
    density_switch = False          # 밀도 추출
    headway_switch = False          # 차두간격 추출

    # 트래킹 파라미터 설정
    test_Video = 'DJI_0167' # 테스트 영상 이름
    exp_num = '20210608' # 실험 이름

    weights_path = 'yolov5/train_result/20210601/weights/best.pt' # 사용할 weights (Yolov5 학습결과로 나온 웨이트 사용)
    test_Video_path = 'input_video/' + test_Video + '.MP4'  # 테스트할 영상 경로 입력
    output_path = 'output_folder/' + test_Video + '_' + exp_num  # 실험결과 저장 경로

    img_size = 800 # 이미지 사이즈(default : 640) : 이미지의 크기를 조절(resizing)하여 검출하도록 만듦, 크면 클수록 검지율이 좋아지지만 FPS가 낮아짐
    conf_thres = 0.4  # 신뢰도 문턱값(default : 0.4) : 해당 수치 중복도 이상은 제거, Yolov5 학습결과(F1_curve.png) 보고 설정 But. 보통 경험적으로 설정
    iou_thres = 0.5  # iou 문턱값(default : 0.5) : 검출 박스의 iou(교집합) 정도
    classes_type = [0, 1, 2] # 데이터셋 및 학습된 모델 클래스 종류

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
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-vid', action='store_false',
                        help='display results')
    parser.add_argument('--save-vid', action='store_true', default=output_path,
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int,
                        default=classes_type, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument("--yolo_swch", default=yolo_switch, help='Yolo on & off')
    parser.add_argument("--deepsort_swch", default=deepsort_switch, help='DeepSORT on & off')
    parser.add_argument("--vehtrk_swch", default=VehTrack_switch, help='Vehicle Track on & off')
    parser.add_argument("--speed_swch", default=speed_switch, help='Speed on & off')
    parser.add_argument("--volume_swch", default=volume_switch, help='Volume on & off')
    parser.add_argument("--line_swch", default=line_switch, help='Line on & off')
    parser.add_argument("--density_swch", default=density_switch, help='Density on & off')
    parser.add_argument("--headway_swch", default=headway_switch, help='Headway on & off')
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    if speed_switch or volume_switch or line_switch:
        test_Video_folder = 'img_matching/control_img/' + test_Video
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




    with torch.no_grad():
        detect(args)
