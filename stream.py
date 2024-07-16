# Uses yolov5 from
# https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov5/python/yolov5.py

import cv2
import argparse
import time
import numpy as np

from yolov5 import setup_model, post_process, draw

IMG_SIZE = (640, 640)  # (width, height)

def process_video(args):
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the GStreamer pipeline for streaming
    gst_str = f'appsrc ! videoconvert ! videoscale ! videorate ! video/x-raw,framerate=30/1 ! vp8enc deadline=1 ! rtpvp8pay ! udpsink host={args.host_ip} port=5000'

    # Open the GStreamer pipeline
    out = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, 30, (int(cap.get(3)), int(cap.get(4))), True)

    model, platform = setup_model(args)

    # Variables for FPS calculation
    prev_time = time.time()
    fps = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w, _ = frame.shape
        img = cv2.resize(frame, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # preprocee if not rknn model
        if platform in ['pytorch', 'onnx']:
            input_data = img.transpose((2,0,1))
            input_data = input_data.reshape(1,*input_data.shape).astype(np.float32)
            input_data = input_data/255.
        else:
            input_data = img
            input_data = input_data.reshape(1, *input_data.shape)


        outputs = model.run([input_data])

        boxes, classes, scores = post_process(outputs, anchors)

        # Calculate FPS
        current_time = time.time()
        elapsed_time = current_time - prev_time
        fps = 1 / elapsed_time
        prev_time = current_time

        # Display FPS on frame
        cv2.putText(frame, f'FPS: {fps:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        if boxes is not None:
            ratio = (orig_h / IMG_SIZE[1], orig_w / IMG_SIZE[0])
            boxes[:, [0, 2]] *= ratio[1]
            boxes[:, [1, 3]] *= ratio[0]
            draw(frame, boxes, scores, classes)

        # Write the frame to the GStreamer pipeline
        out.write(frame)


    cap.release()
    if args.save:
        out.release()
    if args.display:
        cv2.destroyAllWindows()



def main(args):
    if args.video_path:
        process_video(args)
    else:
        print("No video file provided")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Object detection on video")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
    parser.add_argument("--display", action='store_true', help="Display the video")
    parser.add_argument("--save", action='store_true', help="Save the processed video")
    parser.add_argument("--save_path", type=str, default='output.avi', help="Path to save the processed video")
    parser.add_argument("--target", type=str, default='rk3588', help="Target device for RKNN model")
    parser.add_argument("--device_id", type=str, default=None, help="Device ID for RKNN model")
    parser.add_argument('--anchors', type=str, default='../model/anchors_yolov5.txt', help='target to anchor file, only yolov5, yolov7 need this param')
    parser.add_argument('--host_ip', type=str, default='192.168.1.102', help='host ip')

    args = parser.parse_args()

    # load anchor
    with open(args.anchors, 'r') as f:
        values = [float(_v) for _v in f.readlines()]
        anchors = np.array(values).reshape(3,-1,2).tolist()
    print("use anchors from '{}', which is {}".format(args.anchors, anchors))

    main(args)