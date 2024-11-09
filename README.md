# UAV-Tracking
我们这里展示了行人重识别模块的网络重组部分；
用户可以自行使用YOLO算法执行目标检测任务
目标检测的模型，我们使用了我们设计的部分，具体可参考Zhao, Yuhui, Ruifeng Yang, Chenxia Guo and Xiaole Chen. “Parallel Space and Channel Attention for Stronger Remote Sensing Object Detection.” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 17 (2024): 2610-2621.
追踪部分我们使用了现有的滤波算法直接使用，结合我们的目标检测和重识别模块一起使用；
----
我们更新了我们的代码，通过main_imgreid.py文件可以进行行人重识别模型的trian和val；
本文使用的公开数据集是cuhk03，因为数据集过大，上传不容易；读者可以在网上自行下载（如：https://aistudio.baidu.com/datasetdetail/86044/0）
或者联系作者获取相关数据集；
rga_module文件展示了我们的重识别网络结构；
----
本文的目标检测部分，可以结合我们的v7部分，（https://github.com/zyh1122/AdaCBAM-YOLO）
这部分展示了目标检测网络的提升方法
其中，检测和追踪使用的数据集Visdrone_2019det 和visdrone_2019MOT，均为开源数据集，
大家可以从网上自从下载(https://github.com/VisDrone/VisDrone-Dataset)，
或者通过联系作者获取；
----
我们将做好的目标检测网络与重识别网络,和现有的追踪算法相结合，实现无人机目标追踪。
yolotrack部分包含了我们提到的追踪器，使用train.py文件可以执行模型的训练
我们已经把对应的重识别部分结构放在了reid_models文件夹
---
