# VideoPose3d with Detectron2

* Start an AWS EC2 instance with DL AMI. 
* Clone this repository and install the requirements
```bash
pip install -r requirements.txt
```
* Clone the [detectron2](https://github.com/BradleyGrantham/detectron2.git) repository 
and follow the installation instructions. 
* Clone the [VideoPose3D](https://github.com/BradleyGrantham/VideoPose3D.git) repository
and follow the installation instructions. 
* Place your test video in the same directory as these 3 repositories.
* Download the keypoint detection model 
```bash
wget https://dl.fbaipublicfiles.com/detectron2/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x/139686956/model_final_5ad38f.pkl
```

* Then we want to generate the keypoints using `detectron2`. 
The following will generate a file called `pose2d.npz`. 
```bash
cd VideoPose3d_with_Detectron2
python detectron_pose_predictor.py /path/to/input_video
```
* Now we have keypoints, we can use the `VideoPose3D` package to generate a sample
video.
```bash
cd ../VideoPose3D/
python run.py -d custom -k /path/to/keypoints -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject detectron2 --viz-action custom --viz-camera 0 --viz-video /path/to/input_video --viz-output output.mp4 --viz-size 6

```