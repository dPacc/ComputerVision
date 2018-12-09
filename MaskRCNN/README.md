# Mask R-CNN for Object Detection and Segmentation

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. 



## Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in `requirements.txt`.


## Installation
1. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
2. Clone this repository
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
4. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)


### Execution

After installation, test out if everthing is working properly by running `demo.ipynb` in the /samples directory.
I used a custom 'doge.jpg' image for this test and it was able to detect and mask the dog.

- Running on Web Camera:
	Run `python visualize_cv2.py` 
	
	One disadvantage is that the fps captured is (1.5 - 4) which is very slow

- Running on Custom Processed Video:
	Run `python process_video.py`

	It would take a lot of time to process the video, like 1 fps on a GPU. But, it would detect it and
	output a file called `videofile_mask.avi`. This file would be the detected version and you can share it 
	anywhere.




