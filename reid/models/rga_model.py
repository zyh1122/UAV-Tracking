# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable

import torchvision
import numpy as np

from .models_utils.rga_branches import RGA_Branch

__all__ = ['resnet50_rga']
WEIGHT_PATH = os.path.join(os.path.dirname(__file__), '../..')+'/weights/pre_train/resnet50-19c8e357.pth'
#WEIGHT_PATH = './weights/pre_train/resnet50-19c8e357.pth'

# ===================
#   Initialization 
# ===================


#---------------
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
			nn.init.constant_(m.weight, 1.0)
			nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		nn.init.normal_(m.weight, std=0.001)
		if m.bias:
			nn.init.constant_(m.bias, 0.0)
			
# ===============
#    RGA Model 
# ===============

class ResNet50_RGA_Model(nn.Module):
	'''
	Backbone: ResNet-50 + RGA modules.
	'''#num_feat=2048 =64*32=2048=输入图像（256，128）卷积（128，64）池化（64，32）
	def __init__(self, pretrained=True, num_feat=2048, height=256, width=128, 
		dropout=0, num_classes=767, last_stride=1, branch_name='rgasc', scale=8, d_scale=8,
		model_path=WEIGHT_PATH):
		super(ResNet50_RGA_Model, self).__init__()
		self.pretrained = pretrained
		self.num_feat = num_feat
		self.dropout = dropout
		self.num_classes = num_classes
		self.branch_name = branch_name
		print ('Num of features: {}.'.format(self.num_feat))
		
		if 'rgasc' in branch_name:
			spa_on=True 
			cha_on=True
		elif 'rgas' in branch_name:
			spa_on=True
			cha_on=False
		elif 'rgac' in branch_name:
			spa_on=False
			cha_on=True
		else:
			raise NameError
		
		self.backbone = RGA_Branch(pretrained=pretrained, last_stride=last_stride, 
						spa_on=spa_on, cha_on=cha_on, height=height, width=width,
						s_ratio=scale, c_ratio=scale, d_ratio=d_scale, model_path=model_path)

		self.feat_bn = nn.BatchNorm1d(self.num_feat)
		self.feat_bn.bias.requires_grad_(False)
		if self.dropout > 0:
			self.drop = nn.Dropout(self.dropout)
		self.cls = nn.Linear(self.num_feat, self.num_classes, bias=False)

		self.feat_bn.apply(weights_init_kaiming)
		self.cls.apply(weights_init_classifier)
#------------------------------------------------------------------------------
		self.conv_4 = nn.Sequential(
			nn.Conv2d(2048 * 4, 2048, 1, stride=1),
			nn.BatchNorm2d(2048),
			nn.ReLU(inplace=True),
			# nn.Conv2d(32,32,3,stride=1,padding=1),
			# nn.BatchNorm2d(32),
			# nn.ReLU(inplace=True),
			#nn.MaxPool2d(kernel_size=(2, 1) , padding=0),
		)
		self.conv_4_b = nn.Sequential(
			nn.Conv2d(2048, 2048, 1, stride=1),
			nn.BatchNorm2d(2048),
			nn.ReLU(inplace=True),
			# nn.Conv2d(32,32,3,stride=1,padding=1),
			# nn.BatchNorm2d(32),
			# nn.ReLU(inplace=True),
			#nn.MaxPool2d(kernel_size=(1,2), padding=0),
		)
		self.MaxPool2d_H = nn.MaxPool2d(kernel_size=(2, 1), stride=1)
		self.MaxPool2d_W = nn.MaxPool2d(kernel_size=(1, 2), stride=1)


	def forward(self, inputs, training=True):
		im_input = inputs[0]
#		print('*')# [16,3,256,128]
#		print(im_input.shape)
		feat_ = self.backbone(im_input)
#		print('**')
#		print(feat_.shape) #[16,2048,16,8]
#		feat_ = F.avg_pool2d(feat_, feat_.size()[2:]).view(feat_.size(0), -1) #（16，2048，8-16） pooling → （8，2048）
#		print('***')
#		print(feat_.shape)	#[16,2048]
#		feat = self.feat_bn(feat_)
		# print('--------------')
		# print(feat.shape)
		# print('****')	#[16,2048]
		# print(feat_.shape)	#[16,2048]
##########cut########################
		stripe_h_4 = int(feat_.size(2) / 4)
		local_4_feat_list = []  # 局部的特征
		rest_4_feat_list = []  # 去了选中的特征，剩余的局部的特征

		for i in range(4):  # 得到6块中每一个的特征 self.num_stripes=6
			local_4_feat = F.max_pool2d(
				feat_[:, :, i * stripe_h_4: (i + 1) * stripe_h_4, :],
				# 每一块是4*w【：，：，4，：】 分高度取特征块  [b，c，0：4.w]；[b，c，4：8.w]等等.  F.maxpool(input,kerner_size(h,w)) input=分割下来的每个特征段，（(stripe_h_6, feat.size(-1)）指的是最大池化的窗口大小。
				(stripe_h_4, feat_.size(-1)))  # pool成1*1的
			#print(local_4_feat.shape)  # 8 2048 1 1

			local_4_feat_list.append(local_4_feat)  # 按顺序得到每一块的特征，append函数是一个常用的方法，用于向列表、集合和字典等数据结构中添加元素
		# ----------------------------------------------------------------------------
		global_4_max_feat = F.max_pool2d(feat_, (feat_.size(2), feat_.size(3)))  # 8 2048 1 1，全局特征  （8，2048，24/24=1,8/8=1） 全局特征
		local_4_feat_all = (local_4_feat_list[0] + local_4_feat_list[1] + local_4_feat_list[2] + local_4_feat_list[
			3] - global_4_max_feat) / 4  # 求【1，1】的和/6

		local_4_feat_cat = torch.cat(
			(local_4_feat_list[0], local_4_feat_list[1], local_4_feat_list[2], local_4_feat_list[3]),
			1)  # 2048拼接， 然后需要pool得到6，
		local_4_feat_cat = self.conv_4(local_4_feat_cat)

		global_feat = local_4_feat_all - local_4_feat_cat
		# --------------------------------------------------------------------------
		for i in range(4):  # 对于每块特征，除去自己之后其他的特征组合在一起

			rest_4_feat_list.append((local_4_feat_list[(i + 1) % 4]  # 论文公式1处的ri  百分号代表取余
									 + local_4_feat_list[(i + 2) % 4]
									 + local_4_feat_list[(i + 3) % 4]) / 3)
		# ------------------------------------------------
		rest_4_feat_0 = rest_4_feat_list[0]
		rest_4_feat_1 = rest_4_feat_list[1]
		rest_4_feat_2 = rest_4_feat_list[2]
		rest_4_feat_3 = rest_4_feat_list[3]

		# ------------------------------
		rest_0 = global_4_max_feat - rest_4_feat_0
		rest_1 = global_4_max_feat - rest_4_feat_1
		rest_2 = global_4_max_feat - rest_4_feat_2
		rest_3 = global_4_max_feat - rest_4_feat_3
		rest_feat = (rest_0 + rest_1 + rest_2 + rest_3)
		rest_feat_cat = torch.cat((rest_0, rest_1, rest_2, rest_3), 2)
		rest_feat_cat = self.conv_4_b(rest_feat_cat) #16，2048，2，1
		rest_feat_cat = self.MaxPool2d_H(rest_feat_cat)

		# -------------------------------------------------------------------------
		# w维度的分段特征
		stripe_w_4 = int(feat_.size(3) / 4)
		local_w4_feat_list = []  # 局部的特征
		rest_w4_feat_list = []  # 去了选中的特征，剩余的局部的特征

		for i in range(4):  # 得到6块中每一个的特征 self.num_stripes=6
			local_w4_feat = F.max_pool2d(
				feat_[:, :, :, i * stripe_w_4: (i + 1) * stripe_w_4], (feat_.size(-2), stripe_w_4))
			# 每一块是4*w【：，：，4，：】 分高度取特征块  [b，c，0：4.w]；[b，c，4：8.w]等等.  F.maxpool(input,kerner_size(h,w)) input=分割下来的每个特征段，（(stripe_h_6, feat.size(-1)）指的是最大池化的窗口大小。# pool成1*1的
#			print(local_4_feat.shape)  # 8 2048 1 1

			local_w4_feat_list.append(local_w4_feat)  # 按顺序得到每一块的特征，append函数是一个常用的方法，用于向列表、集合和字典等数据结构中添加元素
		# ----------------------------------------------------------------------------
		global_w4_max_feat = F.max_pool2d(feat_, (feat_.size(2), feat_.size(3)))  # 8 2048 1 1，全局特征  （8，2048，24/24=1,8/8=1） 全局特征
		local_w4_feat_all = (local_w4_feat_list[0] + local_w4_feat_list[1] + local_w4_feat_list[2] + local_w4_feat_list[
			3] - global_w4_max_feat) / 4  # 求【1，1】的和/6

		local_w4_feat_cat = torch.cat(
			(local_w4_feat_list[0], local_w4_feat_list[1], local_w4_feat_list[2], local_w4_feat_list[3]),
			1)  # 2048拼接， 然后需要pool得到6，
		local_w4_feat_cat = self.conv_4(local_w4_feat_cat)

		global_w_feat = local_w4_feat_all - local_w4_feat_cat
		# --------------------------------------------------------------------------
		for i in range(4):  # 对于每块特征，除去自己之后其他的特征组合在一起

			rest_w4_feat_list.append((local_w4_feat_list[(i + 1) % 4]  # 论文公式1处的ri  百分号代表取余
									  + local_w4_feat_list[(i + 2) % 4]
									  + local_w4_feat_list[(i + 3) % 4]) / 3)
		# ------------------------------------------------
		rest_w4_feat_0 = rest_w4_feat_list[0]
		rest_w4_feat_1 = rest_w4_feat_list[1]
		rest_w4_feat_2 = rest_w4_feat_list[2]
		rest_w4_feat_3 = rest_w4_feat_list[3]

		# ------------------------------
		rest_w0 = global_w4_max_feat - rest_w4_feat_0
		rest_w1 = global_w4_max_feat - rest_w4_feat_1
		rest_w2 = global_w4_max_feat - rest_w4_feat_2
		rest_w3 = global_w4_max_feat - rest_w4_feat_3

		rest_w_feat = (rest_w0 + rest_w1 + rest_w2 + rest_w3)
		# -------------------------------------------------------------------------------
		rest_w_feat_cat = self.conv_4_b(torch.cat((rest_w0, rest_w1, rest_w2, rest_w3), 3)) #16，2048，1，2
		rest_w_feat_cat = self.MaxPool2d_W (rest_w_feat_cat)
		feat_ =rest_feat+ global_feat + rest_w_feat + global_w_feat+rest_w_feat_cat+rest_feat_cat+global_4_max_feat

#------------finish---------------------
		feat_ = F.avg_pool2d(feat_, feat_.size()[2:]).view(feat_.size(0), -1) #（16，2048，8-16） pooling → （8，2048）
		feat = self.feat_bn(feat_)

		if self.dropout > 0:
			feat = self.drop(feat)
			# print('--------------')
			# print(feat.shape)
		if training and self.num_classes is not None:
			cls_feat = self.cls(feat) #（8，767（类别数量））
			# print('*****')	#[16,767]
			# print(cls_feat.shape)
		
		if training:
			return (feat_, feat, cls_feat) # 池化之后就是feat_，feat_池化之后是feat，经过全连接得到cls_feat
		else:
			return (feat_, feat)

def resnet50_rga(*args, **kwargs):
	return ResNet50_RGA_Model(*args, **kwargs)

