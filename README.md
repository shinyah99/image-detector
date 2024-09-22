# image-detector

## Setup
```
sudo apt-get install libopencv-dev
sudo apt-get install python-opencv

sudo apt install libcblas-dev libatlas3-base libilmbase12 libopenexr22 libgstreamer1.0-0 libqtgui4 libqttest4-perl
sudo pip3 install opencv-python picamera
pip3 install numpy --upgrade
pip3 install pandas
```

### Pytorch
#### Upgrade
```
sudo apt-get update
sudo apt-get dist-upgrade

sudo apt install libopenblas-dev libblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools
sudo apt-get install libavutil-dev libavcodec-dev libavformat-dev libswscale-dev
```

####Download
https://github.com/sungjuGit/PyTorch-and-Vision-for-Raspberry-Pi-4B/blob/master/torch-1.8.0a0%2B56b43f4-cp37-cp37m-linux_armv7l.whl
https://github.com/sungjuGit/PyTorch-and-Vision-for-Raspberry-Pi-4B/blob/master/torchvision-0.9.0a0%2B8fb5838-cp37-cp37m-linux_armv7l.whl

```
sudo pip3 install torch-1.8.0a0+56b43f4-cp37-cp37m-linux_armv7l.whl
sudo pip3 install torchvision-0.9.0a0+8fb5838-cp37-cp37m-linux_armv7l.whl
sudo pip3 install einops
```

## Command

### Training
```
python3 online_classification.py -model nn -num_classes 2 -channels 1 -pixels 112
python3 online_classification.py -model nn -num_classes 2 -channels 3 -pixels 112
python3 online_classification.py -model mvit -num_classes 2 -channels 3 -pixels 256

python3 online_classification.py -model nn -num_classes 3 -channels 1 -pixels 112 -epochs 100
python3 online_classification.py -model nn -num_classes 3 -channels 3 -pixels 112 -epochs 100
```

### Face detection
```
python3 detect_face_camera.py
```

### Object detection
```
python3 take_photo.py
```

command: p

### Start detection
```
python3 online_classification.py -model nn -num_classes 3 -channels 1 -pixels 112 -epochs 200
python3 online_classification.py -model nn -num_classes 3 -channels 3 -pixels 112 -epochs 200
python3 online_classification.py -model mvit -num_classes 3 -channels 3 -pixels 256 -epochs 100
```

## Reference
https://qiita.com/AoChoco/items/a09b446460d95d5c9503
https://www.pc-koubou.jp/magazine/19205
https://ichiri.biz/tech/raspberry_pytorch_install/
https://s-edword.hatenablog.com/entry/2018/01/09/000357
https://www.indoorcorgielec.com/resources/raspberry-pi/raspberry-pi-vnc/
https://github.com/chinhsuanwu/mobilevit-pytorch

