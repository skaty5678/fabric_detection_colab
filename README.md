# YOLOv7 Fabric Detection

This repository contains code for training and evaluating the YOLOv7 model on a fabric detection dataset using Roboflow.


### Clone the YOLOv7 repository:
    git clone https://github.com/WongKinYiu/yolov7


### Install the required dependencies:
    cd yolov7
    pip install -r requirements.txt
    pip install roboflow


### Download the fabric detection dataset from Roboflow:
    from roboflow import Roboflow
    rf = Roboflow(api_key="API_key")
    project = rf.workspace("sanjeev-kumar-thakur-e7isq").project("fabric_detection")
    dataset = project.version(3).download("yolov7")

    
*replace API_key with your Roboflow API key and update the project and dataset information accordingly.*


### Download the COCO starting checkpoint:
    cd /content/yolov7
    wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt


### Training
    cd /content/yolov7
    python train.py --batch 16 --epochs 400 --data {dataset.location}/data.yaml --weights 'yolov7_training.pt' --device 0 


### Evaluation
    python detect.py --weights runs/train/exp/weights/best.pt --conf 0.2 --source {dataset.location}/test/images


### Inference
    import glob
    from IPython.display import Image, display
    
    i = 0
    limit = 10000 # max images to print
    
    for imageName in glob.glob('/content/yolov7/runs/detect/exp/*.jpg'):
        if i < limit:
          display(Image(filename=imageName))
          print("\n")
        i = i + 1

        
![fabric2](https://github.com/skaty5678/fabric_detection_colab/assets/88102311/7b2c254f-f687-4436-b058-f0b978e04bca)
![fabric3](https://github.com/skaty5678/fabric_detection_colab/assets/88102311/ca3e03b9-2f48-4805-af20-f6a57ead0b31)




