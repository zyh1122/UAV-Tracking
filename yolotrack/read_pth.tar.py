import torch
import torchvision.models as models

checkpoint = torch.load('F:\\python_model\\Track\\Yolov7-tracker-master\\weights\\checkpoint_300.pth.tar')	# 加载模型
checkpoint_2 = torch.load('F:\python_model\\Track\Yolov7-tracker-master\weights\ckpt.t7')
print(checkpoint.keys())
for k, v in checkpoint:
    name = k[10:]   # remove `vgg.`，即只取vgg.0.weights的后面几位
    new_state_dict[name] = v
    ssd_net.vgg.load_state_dict(new_state_dict)
print()
print(checkpoint_2.keys())
checkpoint['net_dict'] = checkpoint.pop('state_dict')
print(checkpoint.keys())
print(checkpoint['net_dict'])

# print(checkpoint['epoch'])
# print(checkpoint['best_top1'])
# #print(checkpoint['optimizer'])
# print(checkpoint_2['net_dict'])
# print(checkpoint_2['acc'])
# print(checkpoint_2['epoch'])
