# Models Optimization

## Overview

Convert the YOLOv5 model to ONNX, TFLITE, and RKNN formats and run them on the Orange Pi Board. Aim to achieve maximum FPS with reasonable quality. Stream video with detection boxes, class names, and FPS from the board to your computer using OpenCV gstreamer.

## Model Conversion

### 1. Clone YOLOv5 Repository
```
git clone https://github.com/airockchip/yolov5/
cd yolov5/
pip install -r requirements.txt 
```

### 2. Get the Model (yolov5s.pt)
```
wget 
https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```
### 3. Convert to ONNX
```
python3 export.py --rknpu --weights yolov5s.pt
```
### 4. Convert to TFLITE
```
python3 export.py --weights yolov5s.pt --include tflite --img 320
```

### 5. Convert to RKNN
Use the provided Google Colab notebook [ConvertToRKNN.ipynb](https://colab.research.google.com/drive/1ROlFjNEncSzUSopWnrAUg8FB2xSG_jKl?usp=sharing) for RKNN conversion. For reference, I used the notebook: [Test yolov5-to-rknn without training.ipynb](https://colab.research.google.com/drive/1vqu4Ly7wM7sasTKtg-KXXdp6Ow4_2fNu?usp=sharing#scrollTo=8HU0TlqIWkO9)

## Setup

To set up and use the stream.py and stream_tflite.py scripts for streaming models in ONNX, RKNN, and TFLITE formats, follow these steps:

### 1. Clone the RKNN Model Zoo Repository and Replace `rknn_executor.py`
```
git clone https://github.com/airockchip/rknn_model_zoo cd rknn_model_zoo/py_utils 
wget 
https://raw.githubusercontent.com/violetverve/models_optimization/main/rknn_executor.py -O rknn_executor.py
```

### 2. Add `stream.py` and Models to the Directory
```
cd ../examples/yolov5/python
wget https://raw.githubusercontent.com/violetverve/models_optimization/main/stream.py
```
Get models from the directory: https://github.com/violetverve/models_optimization/tree/main/models

ONNX model is zipped due to its size.

### 3. Clone the YOLO-v5 TFLITE Model Repository and Add `stream_tflite.py`
```
git clone https://github.com/neso613/yolo-v5-tflite-model
cd yolo-v5-tflite-model
wget https://raw.githubusercontent.com/violetverve/models_optimization/main/stream_tflite.py
```
### 4. Handle the Import Error
If you encounter the ‘ImportError: /lib/aarch64-linux-gnu/libgobject-2.0.so.0: undefined symbol: ffi_type_uint32, version LIBFFI_BASE_7.0 error,’ use the following command:

```
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libffi.so.7
```

## Stream videos

### ONNX Execution
```
python3 stream.py --model_path ./yolov5s-i8.onnx --video_path ./videos/cats.mp4 --host_ip 192.168.1.103
```
```
gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp, media=video, encoding-name=VP8, payload=96" ! rtpvp8depay ! vp8dec ! videoconvert ! autovideosink
```
### TFLITE Execution
```
stream_tflite.py --weights ../yolov5s-320.tflite  --source ../videos/cats.mp4 
```
```
gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp,media=video,encoding-name=VP8" ! rtpvp8depay ! vp8dec ! videoconvert ! autovideosink
```
### RKNN Execution
```
python3 stream.py --model_path ./yolov5s-640-i8.rknn --video_path ./videos/cats.mp4 --host_ip 192.168.1.103
```
```
python3 stream.py --model_path ./yolov5s-640-fp.rknn --video_path ./videos/cats.mp4 --host_ip 192.168.1.103
```
```
gst-launch-1.0 udpsrc port=5000 ! application/x-rtp, encoding-name=VP8, payload=96 ! rtpvp8depay ! vp8dec ! videoconvert ! autovideosink
```
## Results

| Format    | FPS |
|-----------|-----|
| ONNX      | 2   |
| TFLITE    | 4   |
| RKNN (fp) | 9   |
| RKNN (i8) | 17  |

The results highlight the advantage of the RKNN format, especially with integer precision, for achieving higher FPS rates on Orangepi 5 Plus compared to ONNX and TFLITE formats.

### Stream Showcase

[Watch the screen recordings here](https://drive.google.com/drive/folders/1CS-Vo95qld6mDV4yIQrdhqwBXeCIfePu?usp=sharing)


## Sources

### GitHub Repositories:
- YOLOv5 Repository: https://github.com/ultralytics/yolov5
- RKNN Model Zoo: https://github.com/airockchip/rknn_model_zoo
- YOLO-v5 TFLITE Model: https://github.com/neso613/yolo-v5-tflite-model
### Google Colab Notebook:
- Convert YOLOv5 to RKNN: Test yolov5-to-rknn without training.ipynb
### Video:
- https://www.youtube.com/watch?v=kC7BIzaIpeA
