# VisionGraph Pose Estimation Example
A visiongraph based pose estimator example with performance measurement. The idea behind this example is to show the performance of the various pose estimation networks implemented in visiongraph.

![pose-example](https://user-images.githubusercontent.com/5220162/161539415-e8b05ed7-dbb7-4aa3-876a-c3b59acb9112.jpg)
*Pose estimation example ([Image Source](https://www.pexels.com/photo/people-posing-at-the-camera-5319298/))*

The software uses a console as UI to render information about the latency and current FPS.

<img width="844" alt="image" src="https://user-images.githubusercontent.com/5220162/161503541-e882b0f3-8428-41a3-96c7-a996d66c39ba.png">

*Terminal UI of vgpose*

## Installation

To get vgpose up and running we recommend to install the following command into a fresh [virtual-env](https://docs.python.org/3/library/venv.html). This will install all the dependencies and creates a script reference with the alias `vgpose` to run the example.

```
pip install git+https://github.com/cansik/visiongraph-pose-estimator.git
```

If you are going to develop based on this project, please install the necessary requirements directly from the `requirements.txt`.

```
pip install -r requirements.txt
```

## Usage

The example is based on visiongraph, a high level computer vision framework. The framework allows for various input sources like webcams, videos, realsense & azure cameras etc. Please refer to the [help](#help) for all the **input** parameters. For a simple webcam input, just run the software without any arguments:

```
vgpose
```

To use a pre-recorded video as the input use the `--channel` parameter:

```
vgpose --channel media/yourposevideo.mp4
```

### Pose Estimator

Visiongraph already implements a variety of pose estimators from Google's MediaPipe, MoveNet and OpenVINO. To set the pose estimator use the `--pose-estimator` parameter. Valid choices are the following:

```
mediapipe
mediapipe-light
mediapipe-heavy

movenet
movenet-192

openpose
openpose-int8
openpose-fp16

aepose
aepose-288-fp16
aepose-448-fp32
```

This is an example on how to use aepose as pose estimator:

```
vgpose --pose-estimator aepose
```

### Performance

By default, the predicted pose estimation is annotated on the input frame and shown with OpenCV's imshow. This can be slow and for real performance testing it is recommended to add the `--performance` argument. This removes the annotation and showing of the frame and shows what the framework is capable of.

## Help

This output has been generated by running `vgpose --help`.

```
usage: vgpose [-h] [--loglevel {critical,error,warning,info,debug}] [-p]
              [--input video-capture,image,realsense]
              [--input-size width height] [--input-fps INPUT_FPS]
              [--input-rotate 90,-90,180] [--input-flip h,v] [--raw-input]
              [--channel CHANNEL] [--input-skip INPUT_SKIP]
              [--input-path INPUT_PATH] [--input-delay INPUT_DELAY] [--depth]
              [--depth-as-input] [-ir] [--exposure EXPOSURE] [--gain GAIN]
              [--white-balance WHITE_BALANCE] [--rs-serial RS_SERIAL]
              [--rs-json RS_JSON] [--rs-play-bag RS_PLAY_BAG]
              [--rs-record-bag RS_RECORD_BAG] [--rs-disable-emitter]
              [--rs-bag-offline]
              [--rs-filter decimation,spatial,temporal,hole-filling [decimation,spatial,temporal,hole-filling ...]]
              [--rs-color-scheme Jet,Classic,WhiteToBlack,BlackToWhite,Bio,Cold,Warm,Quantized,Pattern]
              [--pose-estimator mediapipe,mediapipe-light,mediapipe-heavy,movenet,movenet-192,openpose,openpose-int8,openpose-fp16,aepose,aepose-288-fp16,aepose-448-fp32]

Example Pipeline

optional arguments:
  -h, --help            show this help message and exit
  --loglevel {critical,error,warning,info,debug}
                        Provide logging level. Example --loglevel debug,
                        default=warning
  -p, --performance     Enable performance profiling (no UI).

input provider:
  --input video-capture,image,realsense
                        Image input provider, default: video-capture.
  --input-size width height
                        Requested input media size.
  --input-fps INPUT_FPS
                        Requested input media framerate.
  --input-rotate 90,-90,180
                        Rotate input media.
  --input-flip h,v      Flip input media.
  --raw-input           Skip automatic input conversion to 3-channel image.
  --channel CHANNEL     Input device channel (camera id, video path, image
                        sequence).
  --input-skip INPUT_SKIP
                        If set the input will be skipped to the value in
                        milliseconds.
  --input-path INPUT_PATH
                        Path to the input image.
  --input-delay INPUT_DELAY
                        Input delay time (s).
  --depth               Enable RealSense depth stream.
  --depth-as-input      Use colored depth stream as input stream.
  -ir, --infrared       Use infrared as input stream.
  --exposure EXPOSURE   Exposure value (usec) for depth camera input (disables
                        auto-exposure).
  --gain GAIN           Gain value for depth input (disables auto-exposure).
  --white-balance WHITE_BALANCE
                        White-Balance value for depth input (disables auto-
                        white-balance).
  --rs-serial RS_SERIAL
                        RealSense serial number to choose specific device.
  --rs-json RS_JSON     RealSense json configuration to apply.
  --rs-play-bag RS_PLAY_BAG
                        Path to a pre-recorded bag file for playback.
  --rs-record-bag RS_RECORD_BAG
                        Path to a bag file to store the current recording.
  --rs-disable-emitter  Disable RealSense IR emitter.
  --rs-bag-offline      Disable realtime bag playback.
  --rs-filter decimation,spatial,temporal,hole-filling [decimation,spatial,temporal,hole-filling ...]
                        RealSense depth filter.
  --rs-color-scheme Jet,Classic,WhiteToBlack,BlackToWhite,Bio,Cold,Warm,Quantized,Pattern
                        Color scheme for depth map, default: WhiteToBlack.

pose estimator:
  --pose-estimator mediapipe,mediapipe-light,mediapipe-heavy,movenet,movenet-192,openpose,openpose-int8,openpose-fp16,aepose,aepose-288-fp16,aepose-448-fp32
                        Pose estimator, default: mediapipe.
```

## About
MIT License - Copyright (c) 2022 Florian Bruggisser
