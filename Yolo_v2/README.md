# Yolo-V2

AKA Yolo-9000, it is a high-speed object-detection algorithm that can detect OVER 9000! object categories.

Yolo was initially written in a framework called "Darknet"(completely written in C) and C is not user-friendly,
but luckily someone has created a Tensorflow version of it called "Darkflow".


## Requirements

Download the zip from: https://github.com/thtrieu/darkflow

Extract the zip in you project folder. 
Open you terminal from the terminal and type:

`python setup.py build_ext --inplace`

This should start building your library.

The next step is to download the Yolo-V2 608x608 weights
Click here: https://pjreddie.com/darknet/yolo/

Download the Yolo-V2 weight. Now, create a new folder in the darkflow-master named "bin".
Copy the weights file into this directory.

### To Render a Custom Video

Place a mp4 you want to process in the main directory as `videofile.mp4`

Activate your Tensorflow environment(if you have one). Then run the following command

#### For CPU

```
python flow --model cfg/yolo.cfg --load bin/yolov2.weights --demo videofile.mp4 --saveVideo

```

#### For GPU

```
python flow --model cfg/yolo.cfg --load bin/yolov2.weights --demo videofile.mp4 --gpu=1.0 --saveVideo

```

I only have a CPU, the process speed is about 1.25 fps.
With a GPU, its supposed to hit 15 fps(GTX 1070), my bad xD


### To Process Image in Python


Here we'll be processing static images to add the Box and a label.

Check the `yoloCustomImage.ipynb`


### To Process a Video

Yolo can process about 20 fps on a GPU, so if the video is going to be 60 fps, consider downsampling it to make it look smoother.
Again, if you only have a CPU, it processes only 1.2 fps even after downsampling.

Run `python downsample_video.py`

Then run: `python processing_video.py`


### Real Time Webcam Detection

Run `python webcam_video.py`

The results on a CPU is laggy, should do well on a GPU


### Custom Object Detection

The steps involved are: Collecting data, Annotate data and then train it.

#### Step-1: -

In order to get a good model, its going to come down to how many images you have, the quality of the images and how well you annotate them.
If you only have less than 100 images and poorly annotated, dont expect the model to detect anything.

You need atleast a 1000 images to build a good model.

I have created a new folder called `/new_model_data` which contains the `get_images.py` file

Download images from the web by replacing the `search_name` with your custom object. Then run `python get_images.py`


#### Step-2: -

Annotating can be done with `matplolib.widgets import RectangleSelector`


#### Step-3: -

Train your image by following the steps in Darkflow repository
