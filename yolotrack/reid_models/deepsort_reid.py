"""
file for DeepSORT Re-ID model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import logging
import torchvision.transforms as transforms
import os

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

WEIGHT_PATH = 'F:\\python_model\\Yolov7-tracker-master\\weights\\pre_train\\resnet50-19c8e357.pth'

class RGA_Branch(nn.Module):
    def __init__(self, pretrained=True, last_stride=1, block=Bottleneck, layers=[3, 4, 6, 3],
                 spa_on=True, cha_on=True, s_ratio=8, c_ratio=8, d_ratio=8, height=256, width=128,
                 model_path=WEIGHT_PATH):
        super(RGA_Branch, self).__init__()

        self.in_channels = 64

        # Networks
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

        # RGA Modules
        self.rga_att1 = RGA_Module(256, (height // 4) * (width // 4), use_spatial=spa_on, use_channel=cha_on,
                                   cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
        self.rga_att2 = RGA_Module(512, (height // 8) * (width // 8), use_spatial=spa_on, use_channel=cha_on,
                                   cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
        self.rga_att3 = RGA_Module(1024, (height // 16) * (width // 16), use_spatial=spa_on, use_channel=cha_on,
                                   cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
        self.rga_att4 = RGA_Module(2048, (height // 16) * (width // 16), use_spatial=spa_on, use_channel=cha_on,
                                   cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)

        # Load the pre-trained model weights
        if pretrained:
            self.load_specific_param(self.conv1.state_dict(), 'conv1', model_path)
            self.load_specific_param(self.bn1.state_dict(), 'bn1', model_path)
            self.load_partial_param(self.layer1.state_dict(), 1, model_path)
            self.load_partial_param(self.layer2.state_dict(), 2, model_path)
            self.load_partial_param(self.layer3.state_dict(), 3, model_path)
            self.load_partial_param(self.layer4.state_dict(), 4, model_path)

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def load_partial_param(self, state_dict, model_index, model_path):
        param_dict = torch.load(model_path)
        for i in state_dict:
            try:
                key = 'layer{}.'.format(model_index) + i
                state_dict[i].copy_(param_dict[key])
            except:
                continue
        del param_dict

    def load_specific_param(self, state_dict, param_name, model_path):
        param_dict = torch.load(model_path)
        for i in state_dict:
            try:
                key = param_name + '.' + i
                state_dict[i].copy_(param_dict[key])
            except:
                continue
        del param_dict

    def forward(self, x):
#        print(x.shape)
        x = self.conv1(x)
#        print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
#        print(x.shape)
        x = self.layer1(x)
#        print(x.shape)
        x = self.rga_att1(x)  # 最核心的地方，空间信息和通道信息
#        print(x.shape)
        x = self.layer2(x)
#        print(x.shape)
        x = self.rga_att2(x)
#        print(x.shape)

        x = self.layer3(x)
#        print(x.shape)
        x = self.rga_att3(x)
#        print(x.shape)

        x = self.layer4(x)
#        print(x.shape)
        x = self.rga_att4(x)
#        print(x.shape)

        return x


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out), ]
    return nn.Sequential(*blocks)

#######

#######num_classes=751（Marke-1501 训练集的文件夹个数）

# class Net(nn.Module):
#     def __init__(self, num_classes=1858, reid=False):
#         super(Net, self).__init__()
#         # 3 128 64
#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             # nn.Conv2d(32,32,3,stride=1,padding=1),
#             # nn.BatchNorm2d(32),
#             # nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, 2, padding=1),
#         )
#         # 32 64 32
#         self.layer1 = make_layers(64, 64, 2, False)
#         # 32 64 32
#         self.layer2 = make_layers(64, 128, 2, True)
#         # 64 32 16
#         self.layer3 = make_layers(128, 256, 2, True)
#         # 128 16 8
#         self.layer4 = make_layers(256, 512, 2, True)
#         # 256 8 4
#         self.avgpool = nn.AvgPool2d((8, 4), 1)
#         # 256 1 1
#         self.reid = reid
#         self.classifier = nn.Sequential(
#             nn.Linear(2048, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(256, num_classes),
#         )
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         # B x 128
#         if self.reid:
#             x = x.div(x.norm(p=2, dim=1, keepdim=True))
#             return x
#         # classifier
#         x = self.classifier(x)
#         return x

#WEIGHT_PATH = 'F:\\python_model\\Yolov7-tracker-master\\weights\\pre_train\\resnet50-19c8e357.pth'
class Net(nn.Module):
    def __init__(self, num_classes=1858, reid=False,pretrained=True, last_stride=1,  height=256, width=128, branch_name='rgasc',scale=8, d_scale=8): #num_classes代表的是train中文件夹的个数，也就是追踪的个数
        super(Net, self).__init__()
        # 3 128 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        if 'rgasc' in branch_name:
            spa_on = True
            cha_on = True
        elif 'rgas' in branch_name:
            spa_on = True
            cha_on = False
        elif 'rgac' in branch_name:
            spa_on = False
            cha_on = True
        else:
            raise NameError

        self.backbone = RGA_Branch(pretrained=pretrained, last_stride=last_stride,
                                   spa_on=spa_on, cha_on=cha_on, height=height, width=width,
                                   s_ratio=scale, c_ratio=scale, d_ratio=d_scale, model_path='F:\\python_model\\Yolov7-tracker-master\\weights\\pre_train\\resnet50-19c8e357.pth')# model_path='F:\\python_model\\Yolov7-tracker-master\\weights\\pre_train\\resnet50-19c8e357.pth'
        # 32 64 32
        self.layer1 = make_layers(64, 64, 2, False)
        # 32 64 32
        self.layer2 = make_layers(64, 128, 2, True)
        # 64 32 16
        self.layer3 = make_layers(128, 256, 2, True)
        # 128 16 8
        self.layer4 = make_layers(256, 512, 2, True)
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        # 256 1 1
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(2048*4, 2048, 1, stride=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            #nn.MaxPool2d(3, 2, padding=1),
        )

    def forward(self, x):
#        print(x.shape)  # [16,3,256,128] 3*256*128=983，304
        x = self.backbone(x)
#            print(x.shape)  # [16,2048,16,8]          2048*16*8= 32*128*64=262，144
#         stripe_h_4 = int(x.size(2) / 4)
#         local_4_feat_list = []  # 局部的特征
#         rest_4_feat_list = []  # 去了选中的特征，剩余的局部的特征
#
#         for i in range(4):  # 得到6块中每一个的特征 self.num_stripes=6
#             local_4_feat = F.max_pool2d(
#                 x[:, :, i * stripe_h_4: (i + 1) * stripe_h_4, :],
#                     # 每一块是4*w【：，：，4，：】 分高度取特征块  [b，c，0：4.w]；[b，c，4：8.w]等等.  F.maxpool(input,kerner_size(h,w)) input=分割下来的每个特征段，（(stripe_h_6, feat.size(-1)）指的是最大池化的窗口大小。
#                 (stripe_h_4, x.size(-1)))  # pool成1*1的
# #            print(local_4_feat.shape)  # 8 2048 1 1
#
#             local_4_feat_list.append(local_4_feat)  # 按顺序得到每一块的特征，append函数是一个常用的方法，用于向列表、集合和字典等数据结构中添加元素
#             # ----------------------------------------------------------------------------
#         global_4_max_feat = F.max_pool2d(x, (x.size(2), x.size(3)))  # 8 2048 1 1，全局特征  （8，2048，24/24=1,8/8=1） 全局特征
#         local_4_feat_all = (local_4_feat_list[0] + local_4_feat_list[1] + local_4_feat_list[2] + local_4_feat_list[3] - global_4_max_feat) / 4  # 求【1，1】的和/6
#
#         local_4_feat_cat = torch.cat((local_4_feat_list[0], local_4_feat_list[1], local_4_feat_list[2], local_4_feat_list[3]),1)  # 2048拼接， 然后需要pool得到6，
#         local_4_feat_cat = self.conv_4(local_4_feat_cat)
#
#         global_feat = local_4_feat_all - local_4_feat_cat
#             # --------------------------------------------------------------------------
#         for i in range(4):  # 对于每块特征，除去自己之后其他的特征组合在一起
#
#             rest_4_feat_list.append((local_4_feat_list[(i + 1) % 4]  # 论文公式1处的ri  百分号代表取余
#                                          + local_4_feat_list[(i + 2) % 4]
#                                          + local_4_feat_list[(i + 3) % 4]) / 3)
#             # ------------------------------------------------
#         rest_4_feat_0 = rest_4_feat_list[0]
#         rest_4_feat_1 = rest_4_feat_list[1]
#         rest_4_feat_2 = rest_4_feat_list[2]
#         rest_4_feat_3 = rest_4_feat_list[3]
#
#             # ------------------------------
#         rest_0 = global_4_max_feat - rest_4_feat_0
#         rest_1 = global_4_max_feat - rest_4_feat_1
#         rest_2 = global_4_max_feat - rest_4_feat_2
#         rest_3 = global_4_max_feat - rest_4_feat_3
#         rest_feat = (rest_0 + rest_1 + rest_2 + rest_3)
#         # -------------------------------------------------------------------------
#         # w维度的分段特征
#         stripe_w_4 = int(x.size(3) / 4)
#         local_w4_feat_list = []  # 局部的特征
#         rest_w4_feat_list = []  # 去了选中的特征，剩余的局部的特征
#
#         for i in range(4):  # 得到6块中每一个的特征 self.num_stripes=6
#             local_w4_feat = F.max_pool2d( x[:, :, :, i * stripe_w_4: (i + 1) * stripe_w_4], (x.size(-2), stripe_w_4))
#                 # 每一块是4*w【：，：，4，：】 分高度取特征块  [b，c，0：4.w]；[b，c，4：8.w]等等.  F.maxpool(input,kerner_size(h,w)) input=分割下来的每个特征段，（(stripe_h_6, feat.size(-1)）指的是最大池化的窗口大小。# pool成1*1的
#             #print(local_4_feat.shape)  # 8 2048 1 1
#
#             local_w4_feat_list.append(local_w4_feat)  # 按顺序得到每一块的特征，append函数是一个常用的方法，用于向列表、集合和字典等数据结构中添加元素
#             # ----------------------------------------------------------------------------
#         global_w4_max_feat = F.max_pool2d(x, (x.size(2), x.size(3)))  # 8 2048 1 1，全局特征  （8，2048，24/24=1,8/8=1） 全局特征
#         local_w4_feat_all = (local_w4_feat_list[0] + local_w4_feat_list[1] + local_w4_feat_list[2] +local_w4_feat_list[3] - global_w4_max_feat) / 5  # 求【1，1】的和/6
#
#         local_w4_feat_cat = torch.cat((local_w4_feat_list[0], local_w4_feat_list[1], local_w4_feat_list[2], local_w4_feat_list[3]),1)  # 2048拼接， 然后需要pool得到6，
#         local_w4_feat_cat = self.conv_4(local_w4_feat_cat)
#
#         global_w_feat = local_w4_feat_all - local_w4_feat_cat
# # --------------------------------------------------------------------------
#         for i in range(4):  # 对于每块特征，除去自己之后其他的特征组合在一起
#             rest_w4_feat_list.append((local_w4_feat_list[(i + 1) % 4]  # 论文公式1处的ri  百分号代表取余
#                                           + local_w4_feat_list[(i + 2) % 4]
#                                           + local_w4_feat_list[(i + 3) % 4]) / 3)
#             # ------------------------------------------------
#         rest_w4_feat_0 = rest_w4_feat_list[0]
#         rest_w4_feat_1 = rest_w4_feat_list[1]
#         rest_w4_feat_2 = rest_w4_feat_list[2]
#         rest_w4_feat_3 = rest_w4_feat_list[3]
# # ------------------------------
#         rest_w0 = global_w4_max_feat - rest_w4_feat_0
#         rest_w1 = global_w4_max_feat - rest_w4_feat_1
#         rest_w2 = global_w4_max_feat - rest_w4_feat_2
#         rest_w3 = global_w4_max_feat - rest_w4_feat_3
#
#         rest_w_feat = (rest_w0 + rest_w1 + rest_w2 + rest_w3)
#             # -------
#         x = rest_feat + global_feat + rest_w_feat + global_w_feat
            # -------------------
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=torch.device(self.device))['net_dict']
        print(state_dict.keys())
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (256, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            try:
                return cv2.resize(im.astype(np.float32)/255., size)
            except:
                print('Error: size in bbox exists zero, ', im.shape)
                exit(0)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(
            0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()

#-----zyh
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


def weights_init_fc(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


class RGA_Module(nn.Module):
    def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=True,  # 是否使用空间or通道注意力机制，两个布尔值
                 cha_ratio=8, spa_ratio=8, down_ratio=8):
        super(RGA_Module, self).__init__()

        self.in_channel = in_channel
        self.in_spatial = in_spatial

        self.use_spatial = use_spatial
        self.use_channel = use_channel

        print('Use_Spatial_Att: {};\tUse_Channel_Att: {}.'.format(self.use_spatial, self.use_channel))

        self.inter_channel = in_channel // cha_ratio  # 内部的通道注意力
        self.inter_spatial = in_spatial // spa_ratio  # 内部的空间注意力

        # Embedding functions for original features
        if self.use_spatial:
            self.gx_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                # BN的目的是使得我们的一批（batch）feature map满足均值为0,方差为1的分布规律，***BatchNorm后是不改变输入的shape的
                nn.ReLU()
            )
        if self.use_channel:
            self.gx_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

        # Embedding functions for relation features
        if self.use_spatial:
            self.gg_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
        if self.use_channel:
            self.gg_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel * 2, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )

        # Networks for learning attention weights
        if self.use_spatial:
            num_channel_s = 1 + self.inter_spatial
            self.W_spatial = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s // down_ratio,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_channel_s // down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_s // down_ratio, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )
        if self.use_channel:
            num_channel_c = 1 + self.inter_channel
            self.W_channel = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_c, out_channels=num_channel_c // down_ratio,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_channel_c // down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_c // down_ratio, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )

        # Embedding functions for modeling relations
        if self.use_spatial:
            self.theta_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
            self.phi_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.theta_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
            self.phi_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

    def forward(self, x):
        b, c, h, w = x.size()
        if self.use_spatial:  # spatial attention
            # print(x.shape)
            theta_xs = self.theta_spatial(x)  # 8 32 64 32
            # print(theta_xs.shape)
            phi_xs = self.phi_spatial(x)  # 8 32 64 32
            # print(phi_xs.shape)
            theta_xs = theta_xs.view(b, self.inter_channel, -1)  # 8 32 64*32
            # print(theta_xs.shape)
            theta_xs = theta_xs.permute(0, 2, 1)  # 8 64*32 32
            # print(theta_xs.shape)
            phi_xs = phi_xs.view(b, self.inter_channel, -1)  # 8 32 64*32
            # print(phi_xs.shape)
            Gs = torch.matmul(theta_xs, phi_xs)  # 8 2048 2048
            # print(Gs.shape)
            Gs_in = Gs.permute(0, 2, 1).view(b, h * w, h, w)  # 8 2048 64 32 调换下顺序
            # print(Gs_in.shape)
            Gs_out = Gs.view(b, h * w, h, w)  # 8 2048 64 32
            # print(Gs_out.shape)
            Gs_joint = torch.cat((Gs_in, Gs_out), 1)  # 8 4096 64 32
            # print(Gs_joint.shape)
            Gs_joint = self.gg_spatial(Gs_joint)  # 8 256 64 32
            # print(Gs_joint.shape)

            g_xs = self.gx_spatial(x)  # 8 32 64 32
            # print(g_xs.shape)
            g_xs = torch.mean(g_xs, dim=1, keepdim=True)  # 8 1 64 32
            # print(g_xs.shape)
            ys = torch.cat((g_xs, Gs_joint), 1)  # 8 257 64 32
            # print(ys.shape)

            W_ys = self.W_spatial(ys)  # 8 1 64 32
            # print(W_ys.shape)
            if not self.use_channel:
                out = F.sigmoid(W_ys.expand_as(x)) * x  # 位置特征，不同特征图，位置相同的
                return out
            else:
                x = F.sigmoid(W_ys.expand_as(x)) * x
        # print(x.shape) # 8 256 64 32
        if self.use_channel:  # channel attention
            xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)  # 8 2048 256 1
            # print(xc.shape)
            theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)  # 8 256 256 ###特征图之间的关系
            # print(theta_xc.shape)
            phi_xc = self.phi_channel(xc).squeeze(-1)  # 8 256 256
            # print(phi_xc.shape)
            Gc = torch.matmul(theta_xc, phi_xc)  # 8 256 256
            # print(Gc.shape)
            Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)  # 8 256 256 1
            # print(Gc_in.shape)
            Gc_out = Gc.unsqueeze(-1)  # 8 256 256 1
            # print(Gc_out.shape)
            Gc_joint = torch.cat((Gc_in, Gc_out), 1)  # 8 512 256 1
            # print(Gc_joint.shape)
            Gc_joint = self.gg_channel(Gc_joint)  # 8 32 256 1
            # print(Gc_joint.shape)

            g_xc = self.gx_channel(xc)  # 8 256 256 1
            # print(g_xc.shape)
            g_xc = torch.mean(g_xc, dim=1, keepdim=True)  # 8 1 256 1
            # print(g_xc.shape)
            yc = torch.cat((g_xc, Gc_joint), 1)  # 8 33 256 1
            # print(yc.shape)
            W_yc = self.W_channel(yc).transpose(1, 2)  # 8 256 1 1 得到权重分配
            # print(W_yc.shape)
            out = F.sigmoid(W_yc) * x
            # print(out.shape)

            return out

