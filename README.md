# UAV_MOT_AutoGeorefencing

![](sample.gif)


## Introduction

본 문서는 드론 항공영상 내 차량 객체를 검지, 추적하고 자동으로 지오레퍼런싱을 수행하여 영상 내 개별차량의 지리적 위치좌표를 추출하는 레파지토리입니다.
YOLOv5와 DeepSort를 사용하여 Multiple Object Tracking을 수행하며, 개발된 Auto Georeferencing Framework 모델을 활용하여 드론 항공영상의 모든 프레임에 대해 자동 지오레퍼런싱을 수행합니다.
그 외 옵션에 따른 카메라 캘리브레이션, 차량 객체 검지, 차량 객체 추적, 차량 주행궤적 추출, 차량별 속도 추출, 교통량 추출, 지오레퍼런싱을 수행할 수 있고, 해당 옵션은 `track.py` 에서 선택할 수 있습니다.
본 레파지토리는 PyTorch YOLOv5 (https://github.com/ultralytics/yolov5)과 Yolov5_DeepSort_Pytorch (https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)를 기반으로 만들어졌습니다.


## Requirements

본 문서의 검지 파트는 PyTorch YOLOv5 (https://github.com/ultralytics/yolov5)를 사용합니다. 따라서 해당 문서의 `requirments.txt`를 따른 뒤 아래 코드를 실행하세요.

`pip install -U -r requirements.txt`

도커를 사용하기 위해서는 다음과 같은 요구사항을 따라야합니다.
- `nvidia-docker`
- Nvidia Driver Version >= 440.44
위 요구사항을 충족한 이후 아래 코드를 실행하여 설치하세요.

`pip install -U -r requirements_docker.txt`

또는 본인의 DockerHub에 push된 이미지를 활용하여 이미 구축된 환경을 컨테이너로 활용하세요.

`docker push bok9504/uav_mot_autogeoreferencing:tagname`


## File Structure

파일 구조는 아래와 같습니다.

```
~/.Yolov5_DeepSort_Pytorch-master_v2
├── data
│   ├── data_setting
│   │   ├── calibration_info
│   │   ├── control_point
│   │   ├── counter_point
│   │   └── source_img
│   ├── input_video
│   └── output_folder
├── Georeferencing/image_registration
├── MOT
│   ├── deep_sort_pytorch
│   └── yolov5
└── Pre_processing
```


1. data
- `data` 파일에는 `data_setting`, `input_video`, `output_folder` 폴더가 포함되어 있습니다.
- `data_setting`에는 본 코드의 옵션을 사용하기 위해 필요한 데이터 정보가 포함됩니다. calibration_info를 제외한 나머지 옵션은 `track.py` 코드의 실행 이후 생성 가능합니다.
- `input_video`에는 입력으로 활용할 드론 항공영상이 포함됩니다. `sample_video.MP4` 파일이 포함되어 있습니다.
- `output_folder`에는 `track.py` 코드의 실행 결과물이 자동으로 저장됩니다. 실행 결과물은 드론 항공영상에 대한 비디오 파일과, 데이터가 추출된 텍스트 파일을 포함합니다.

2. Georeferencing/image_registration
- `Georeferencing/image_registration` 폴더에는 지오레퍼런싱을 위한 코드파일이 포함되어 있습니다.
- `feature_matching.py` 코드는 feature matching 기반의 영상정합을 위한 코드입니다. `track.py` 코드 내 옵션 선택을 통하여 descriptor, Matcher, matching_func을 임의적으로 선택할 수 있습니다.
- `update_source_Image.py` 코드는 삼변측량을 통한 GCP의 갱신을 수행하는 코드입니다.

3. MOT
- `MOT` 폴더는 Multiple Object Tracking을 위한 폴더입니다. 기본적으로  PyTorch YOLOv5 (https://github.com/ultralytics/yolov5)과 Yolov5_DeepSort_Pytorch (https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)를 따릅니다.

4. Pre_processing
- `Pre_processing` 폴더는 `./data/data_setting`에 포함되는 데이터를 생성해주는 코드를 포함합니다. 본 코드들은 `track.py` 코드와 연동되어 `track.py`의 실행 시, GUI 프로그램을 통해 `./data/data_setting`에 포함될 데이터를 생성할 수 있습니다.
- `create_*.py` 코드들은 각각 control_point, counter_point, source_img 데이터를 생성하기 위한 코드입니다. `Enter_GeoCoordinates.py` 코드와 연동되어 GUI 기반의 데이터 생성을 수행합니다.
- `camera_calibration.py` 코드는 이용자의 카메라의 보정값을 보유시 카메라 캘리브레이션을 수행하는 코드입니다. 본 레파지토리에 포함된 `./data/data_setting/mavic2_pro.npz`는 본 연구에 사용된 드론인 MAVIC2 PRO 카메라의 보정값입니다.


## How to run

1. track.py 옵션(Function) 세팅
- `track.py` 코드 내 main() 파일에 존재하는 `# Choose Function`에서 이용자가 원하는 옵션을 선택합니다. True로 세팅시 해당옵션을 사용하고, False로 세팅시 해당옵션을 끌 수 있습니다. 각 옵션별로 `./data/data_setting`에 존재하는 데이터가 필요한 경우가 있고 필요없는 경우가 존재합니다. 자세한 내용은 아래 후술됩니다.
- `camera_calibrate_switch` 은 항공영상에 대한 카메라 캘리브레이션을 수행하여 보정된 동영상 자료를 획득할 수 있습니다. 하지만 GCP 설정이 달라 질 수 있음으로 영상정합이 필요한 경우 옵션을 끄는것을 추천합니다.
- `yolo_switch`, `deepsort_switch`는 차량 검지와 추적 중 필요한 옵션을 선택할 수 있습니다. 다만 옵션선택은 동영상 내 표출 여부의 선택이며, 기본적으로 `track.py` 코드는 수행시, 객체 검지와 추적을 자동으로 수행합니다.
- `VehTrack_switch`는 개별 차량 객체의 주행궤적을 표출해주는 옵션입니다.
- `img_registration_switch`는 Source image를 기반으로 GCP를 업데이트 시켜주는 옵션입니다. `speed_switch`, `volume_switch`, `Georeferencing_switch` 옵션을 활성화시키기 위해서는 해당 옵션을 반드시 선택해주어야 합니다. 해당 옵션의 선택시 `./data/data_setting`에 포함되는 `source_img`, `control_point`를 보유해야합니다. `source_img`, `control_point`를 설정하는 방법은 `3. 전처리 데이터 생성`에서 자세히 설명합니다.
- `speed_switch`는 개별 차량 객체에 대한 속도 정보를 추출하는 옵션입니다. 해당 옵션의 활성화 시 `output_folder`에 생성되는 비디오 파일에는 차량 객체별 속도정보가 표출되며, 텍스트 파일에는 개별 차량의 프레임별 속도 데이터가 추출됩니다.
- `volume_switch`는 교통량 검지기(counter)를 통과하는 차량을 카운팅하는 옵션입니다. 카운팅 결과는 비디오파일에 실시간으로 표출됩니다. 해당 옵션을 활성화하기 위해서는 `./data/data_setting`에 포함되는 `counter_point`를 보유해야합니다. `counter_point`를 설정하는 방법은 `4. 전처리 데이터 생성`에서 자세히 설명합니다.
- `Georeferencing_switch`는 동영상의 모든 프레임에 자동으로 지오레퍼런싱을 수행하는 옵션입니다. 해당 옵션을 활성화할 시 생성되는 텍스트 파일에 개별 차량의 프레임별 경위도좌표가 추출됩니다. 

2. track.py 파라미터 세팅
- `track.py` 코드 내 main() 파일에 존재하는  `# Setting Parameters`에서 이용자가 원하는 옵션을 선택합니다. 테스트할 영상의 이름, 실험의 이름, 사용할 YOLO weight 파일의 경로, YOLO 검지 이미지 사이즈, 검지 confidence, iou thres 등을 설정해줍니다.

3. `track.py` 실행

cmd 창에 아래 코드를 실행합니다.

```bash
python track.py
```

4. 전처리 데이터 생성
- 지오레퍼런싱과 영상정합을 위해 필요한 `source_img`, `control_point`, `counter_point`를 생성하는 파트입니다. `track.py` 실행하면 아래와 같은 명령어가 출력됩니다.

```
If you wanna create new source image, Write yes :
If you wanna create new control point, Write yes :
If you wanna create new counter point, Write yes :
```

- 만약 각 `source_img`, `control_point`, `counter_point`를 새롭게 생성하고 싶으면 'yes'를 입력하시면 됩니다. 'yes'를 입력 시 해당 데이터를 생성하는 GUI 프로그램이 생성됩니다. 만약 'yes'를 제외한 다른 버튼을 입력하면 기존에 Sample_video가 가지고 있는 Source image 와 Control Point, Counter Point를 사용합니다. 'yes'를 선택시 활성화 되는 데이터를 생성하는 GUI 프로그램에 대한 설명은 아래 후술됩니다.

 1) Source image
 - `source_img`를 생성하는 GUI 프로그램입니다. Source image는 영상정합을 위해 필요한 기준 이미지로 드론 영상 첫 프레임을 기준으로 일부 이미지를 잘라 생성합니다. Source image는 프레임 이미지 내 특징점이 많이 존재하는 구역에 3개, 방사형으로 위치하도록 선택하는 것을 권장합니다.
 - 마우스로 화면을 클릭하면 클릭한 구간을 중심으로 바운딩 박스가 생성됩니다. 위와 같은 절차를 총 3번 반복하여 수행한 뒤 's' 버튼을 입력하면 선택한 Source image 세장이 저장됩니다.
 - 만약 선택한 Source image를 변경하고 싶은 경우 'r' 버튼을 입력하면 최근에 선택된 Source image가 삭제됩니다.
 - 선택을 하지않고 GUI 프로그램을 끝내고 싶은 경우 'q' 버튼을 입력합니다. 해당 경우 Source image를 새롭게 저장하지 않고 기존에 존재하는 Source image를 활용합니다.

 2) Control Point
 - `control_point`는 지오레퍼런싱을 위한 GCP로 이미지상의 위치정보에 경위도좌표를 부여하기위하여 활용됩니다. 드론 항공영상 내에서 정확한 위치정보를 알고 있는 지점을 이미지 내에서 선택하여 경위도좌표를 매칭시켜줍니다. GCP 좌표는 도로 전체를 커버하도록 골구로 분산하여 최소 6개에서 10개 좌표의 선택을 권장합니다.
 - Source image 와 마찬가지로 GUI 프로그램이 활성화 됩니다. 마우스 휠을 사용하여 확대를 하여 정확하게 경위도 좌표를 알고있는 지점을 GCP로 선정하고 선택해줍니다.
 - 선택을 수행하면 `Enter GeoCoordinates`라는 윈도우 화면이 활성화 됩니다. 해당 윈도우에 존재하는 `longitude : ` 과 `latitude  : `란에 각각 위도와 경도를 입력한 후 Enter 버튼을 클릭합니다.
 - 다음과 같은 절차를 GCP 개수 만큼 반복하여줍니다. 선택된 GCP를 저장하기 위해서는 Source image와 같이 's' 버튼을, 삭제는 'r' 버튼을, GUI 프로그램 종료는 'q' 버튼을 입력합니다.

 3) Counter Point
 - `counter_point`는 


## Description

본 문서의 학습데이터와 알고리즘, 속도, 교통량 데이터 정확도에 대한 자세한 정보는 아래 논문에서 확인할 수 있습니다.

```
다중객체추적 알고리즘을 활용한 드론 항공영상 기반 미시적 교통데이터 추출 (2021)
    Microscopic Traffic Parameters Estimation from UAV Video Using Multiple Object Tracking of Deep Learning-based (2021)

    Bokyung Jung, Boogi Park, Sunghyuk Seo, Sanghoon Bae

    The Journal of The Korea Institute of Intelligent Transport Systems, vol.20, no.5, pp.83~99
    https://doi.org/10.12815/kits.2021.20.5.83
```

지오레퍼런싱을 활용한 연구결과는 현재 게재절차 진행 중인 `드론영상 기반 자율주행차량용 LDM 콘텐츠 구축 : 차로변경 행태를 중심으로` 논문에서 확인할 수 있습니다. 추후 업데이트 하도록 하겠습니다.

## Tracking

Tracking can be run on most video formats

```bash
python3 track.py --source ...
```

- Video:  `--source file.mp4`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`

MOT compliant results can be saved to `inference/output` by 

```bash
python3 track.py --source ... --save-txt
```

## Other information

For more detailed information about the algorithms and their corresponding lisences used in this project access their official github implementations.

