# Change detection head

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
from model.cd_modules.psp import _PSPModule
from model.cd_modules.se import ChannelSpatialSELayer

def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    '''
    Get the number of input layers to the change detection head.
    获取中间的双时特征的通道数
    '''
    in_channels = 0
    for scale in feat_scales:
        if scale < 3: #256 x 256
            in_channels += inner_channel*channel_multiplier[0]
        elif scale < 6: #128 x 128
            in_channels += inner_channel*channel_multiplier[1]
        elif scale < 9: #64 x 64
            in_channels += inner_channel*channel_multiplier[2]
        elif scale < 12: #32 x 32
            in_channels += inner_channel*channel_multiplier[3]
        elif scale < 15: #16 x 16
            in_channels += inner_channel*channel_multiplier[4]
        else:
            print('Unbounded number for feat_scales. 0<=feat_scales<=14') 
    return in_channels

class AttentionBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(inplace=True),
            ChannelSpatialSELayer(num_channels=dim_out, reduction_ratio=2)
        )

    def forward(self, x):
        return self.block(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim*len(time_steps), dim, 1)# 变化检测时要同级特征合并，所以Block的输入维度为dim*len(time_steps)
            if len(time_steps)>0
            else None,
            nn.ReLU(inplace=True)
            if len(time_steps)>0
            else None,
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class cd_head(nn.Module):
    '''
    Change detection head (version 2).
    '''
    # feat_scales 表示要使用哪层特征用于变化检测，因为Unet返回的编码特征有15层，解码特征有20层，因此最大为14。
    # feat_scales=[2, 5, 8, 11, 14], inner_channel=128, channel_multiplier=[1,2,4,8,8],time_steps=[50,100,400]
    def __init__(self, feat_scales, out_channels=2, inner_channel=None, channel_multiplier=None, img_size=256, time_steps=None):
        super(cd_head, self).__init__()

        # Define the parameters of the change detection head
        feat_scales.sort(reverse=True)
        self.feat_scales    = feat_scales
        self.in_channels    = get_in_channels(feat_scales, inner_channel, channel_multiplier)# 这个没用到
        self.img_size       = img_size
        self.time_steps     = time_steps# 图片扩散的step数

        # Convolutional layers before parsing to difference head
        self.decoder = nn.ModuleList()
        for i in range(0, len(self.feat_scales)):#[0, 1, 2, 3, 4]
            dim     = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)

            self.decoder.append(
                Block(dim=dim, dim_out=dim, time_steps=time_steps)
            )

            if i != len(self.feat_scales)-1:
                dim_out =  get_in_channels([self.feat_scales[i+1]], inner_channel, channel_multiplier)
                self.decoder.append(
                AttentionBlock(dim=dim, dim_out=dim_out)
            )

        # Final classification head
        clfr_emb_dim = 64
        self.clfr_stg1 = nn.Conv2d(dim_out, clfr_emb_dim, kernel_size=3, padding=1)
        self.clfr_stg2 = nn.Conv2d(clfr_emb_dim, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats_A, feats_B):
        # Decoder
        lvl=0
        for layer in self.decoder:
            if isinstance(layer, Block):# 这里面是分辨率不变的操作，一个stage
                f_A = feats_A[0][self.feat_scales[lvl]]
                f_B = feats_B[0][self.feat_scales[lvl]]
                for i in range(1, len(self.time_steps)):
                    f_A = torch.cat((f_A, feats_A[i][self.feat_scales[lvl]]), dim=1)#每个图片多次加噪声，得到多张图片，将同级特征先按通道合并。
                    f_B = torch.cat((f_B, feats_B[i][self.feat_scales[lvl]]), dim=1)
    
                diff = torch.abs( layer(f_A)  - layer(f_B) )
                if lvl!=0:
                    diff = diff + x
                lvl+=1
            else:
                diff = layer(diff)# 改变通道数，进入下一个stage，统一通道数
                x = F.interpolate(diff, scale_factor=2, mode="bilinear")# 上采样，统一分辨率

        # Classifier
        cm = self.clfr_stg2(self.relu(self.clfr_stg1(x)))

        return cm

    

class cd_head_v2(nn.Module):
    '''
    Change detection head (version 2).
    '''
    # feat_scales 表示要使用哪层特征用于变化检测，因为Unet返回的编码特征有15层，解码特征有20层，因此最大为14。
    # feat_scales=[2, 5, 8, 11, 14], inner_channel=128, channel_multiplier=[1,2,4,8,8],time_steps=[50,100,400]
    def __init__(self, feat_scales, out_channels=2, inner_channel=None, channel_multiplier=None, img_size=256, time_steps=None, enable_amp=False):
        super(cd_head_v2, self).__init__()

        # Define the parameters of the change detection head
        feat_scales.sort(reverse=True)
        self.enable_amp = enable_amp
        self.feat_scales    = feat_scales
        self.in_channels    = get_in_channels(feat_scales, inner_channel, channel_multiplier)# 这个没用到
        self.img_size       = img_size
        self.time_steps     = time_steps# 图片扩散的step数

        # Convolutional layers before parsing to difference head
        self.decoder = nn.ModuleList()
        for i in range(0, len(self.feat_scales)):#[0, 1, 2, 3, 4]
            dim     = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)

            self.decoder.append(
                Block(dim=dim, dim_out=dim, time_steps=time_steps)
            )

            if i != len(self.feat_scales)-1:
                dim_out =  get_in_channels([self.feat_scales[i+1]], inner_channel, channel_multiplier)
                self.decoder.append(
                AttentionBlock(dim=dim, dim_out=dim_out)
            )

        # Final classification head
        clfr_emb_dim = 64
        self.clfr_stg1 = nn.Conv2d(dim_out, clfr_emb_dim, kernel_size=3, padding=1)
        self.clfr_stg2 = nn.Conv2d(clfr_emb_dim, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats_A, feats_B):
        # Decoder
        lvl=0
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            for layer in self.decoder:
                if isinstance(layer, Block):# 这里面是分辨率不变的操作，一个stage
                    f_A = feats_A[0][lvl]
                    f_B = feats_B[0][lvl]
                    for i in range(1, len(self.time_steps)):
                        f_A = torch.cat((f_A, feats_A[i][lvl]), dim=1)#每个图片多次加噪声，得到多张图片，将同级特征先按通道合并。
                        f_B = torch.cat((f_B, feats_B[i][lvl]), dim=1)
        
                    diff = torch.abs( layer(f_A)  - layer(f_B) )
                    if lvl!=0:
                        diff = diff + x
                    lvl+=1
                else:
                    diff = layer(diff)# 改变通道数，进入下一个stage，统一通道数
                    x = F.interpolate(diff, scale_factor=2, mode="bilinear")# 上采样，统一分辨率
            del diff, f_A, f_B

            # Classifier
            cm = self.clfr_stg2(self.relu(self.clfr_stg1(x)))
            if cm.shape[2] != self.img_size:
                cm = F.interpolate(cm, size=(self.img_size,self.img_size), mode='bilinear', align_corners=True)

            return cm
    

class cd_head_semi(nn.Module):
    '''
    Change detection head (version 2).
    '''
    # feat_scales 表示要使用哪层特征用于变化检测，因为Unet返回的编码特征有15层，解码特征有20层，因此最大为14。
    # feat_scales=[2, 5, 8, 11, 14], inner_channel=128, channel_multiplier=[1,2,4,8,8],time_steps=[50,100,400]
    def __init__(self, feat_scales, out_channels=2, inner_channel=None, channel_multiplier=None, img_size=256, time_steps=None, layers_perturb=2):
        super(cd_head_semi, self).__init__()

        # Define the parameters of the change detection head
        feat_scales.sort(reverse=True)
        self.feat_scales    = feat_scales
        self.in_channels    = get_in_channels(feat_scales, inner_channel, channel_multiplier)# 这个没用到
        self.img_size       = img_size
        self.time_steps     = time_steps# 图片扩散的step数
        self.layers_perturb = layers_perturb

        # Convolutional layers before parsing to difference head
        self.decoder = nn.ModuleList()
        for i in range(0, len(self.feat_scales)):
            dim     = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)

            self.decoder.append(
                Block(dim=dim, dim_out=dim, time_steps=time_steps)
            )

            if i != len(self.feat_scales)-1:
                dim_out =  get_in_channels([self.feat_scales[i+1]], inner_channel, channel_multiplier)
                self.decoder.append(
                AttentionBlock(dim=dim, dim_out=dim_out)
            )

        # Final classification head
        clfr_emb_dim = 64
        self.clfr_stg1 = nn.Conv2d(dim_out, clfr_emb_dim, kernel_size=3, padding=1)
        self.clfr_stg2 = nn.Conv2d(clfr_emb_dim, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats_A, feats_B, perturbation=None, o_l = None):
        # Decoder
        lvl=0
        for layer in self.decoder:
            if isinstance(layer, Block):# 这里面是分辨率不变的操作，一个stage
                f_A = feats_A[0][self.feat_scales[lvl]]
                f_B = feats_B[0][self.feat_scales[lvl]]
                for i in range(1, len(self.time_steps)):
                    f_A = torch.cat((f_A, feats_A[i][self.feat_scales[lvl]]), dim=1)#每个图片多次加噪声，得到多张图片，将同级特征先按通道合并。
                    f_B = torch.cat((f_B, feats_B[i][self.feat_scales[lvl]]), dim=1)
                diff = torch.abs( layer(f_A)  - layer(f_B) )
                if lvl<self.layers_perturb and perturbation:
                    diff = perturbation(diff, o_l)
                if lvl!=0:
                    diff = diff + x
                lvl+=1
                
            else:
                diff = layer(diff)# 改变通道数，进入下一个stage，统一通道数
                x = F.interpolate(diff, scale_factor=2, mode="bilinear")# 上采样，统一分辨率

        # Classifier
        cm = self.clfr_stg2(self.relu(self.clfr_stg1(x)))

        return cm
    
class cd_head_semi_v2(nn.Module):
    '''
    Change detection head (version 2).
    '''
    # feat_scales 表示要使用哪层特征用于变化检测，因为Unet返回的编码特征有15层，解码特征有20层，因此最大为14。
    # feat_scales=[2, 5, 8, 11, 14], inner_channel=128, channel_multiplier=[1,2,4,8,8],time_steps=[50,100,400]
    def __init__(self, feat_scales, out_channels=2, inner_channel=None, channel_multiplier=None, 
                 img_size=256, time_steps=None, layers_perturb=2, enable_amp=False):
        super(cd_head_semi_v2, self).__init__()

        # Define the parameters of the change detection head
        feat_scales.sort(reverse=True)
        self.feat_scales    = feat_scales
        self.in_channels    = get_in_channels(feat_scales, inner_channel, channel_multiplier)# 这个没用到
        self.img_size       = img_size
        self.time_steps     = time_steps# 图片扩散的step数
        self.layers_perturb = layers_perturb
        self.enable_amp = enable_amp

        # Convolutional layers before parsing to difference head
        self.decoder = nn.ModuleList()
        for i in range(0, len(self.feat_scales)):#[0, 1, 2, 3, 4]
            dim     = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)

            self.decoder.append(
                Block(dim=dim, dim_out=dim, time_steps=time_steps)
            )

            if i != len(self.feat_scales)-1:
                dim_out =  get_in_channels([self.feat_scales[i+1]], inner_channel, channel_multiplier)
                self.decoder.append(
                AttentionBlock(dim=dim, dim_out=dim_out)
            )

        # Final classification head
        clfr_emb_dim = 64
        self.clfr_stg1 = nn.Conv2d(dim_out, clfr_emb_dim, kernel_size=3, padding=1)
        self.clfr_stg2 = nn.Conv2d(clfr_emb_dim, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats_A, feats_B, perturbation=None, o_l = None):
        # Decoder
        lvl=0
        with torch.cuda.amp.autocast(enabled=self.enable_amp):
            for layer in self.decoder:
                if isinstance(layer, Block):# 这里面是分辨率不变的操作，一个stage
                    f_A = feats_A[0][lvl]
                    f_B = feats_B[0][lvl]
                    for i in range(1, len(self.time_steps)):
                        f_A = torch.cat((f_A, feats_A[i][lvl]), dim=1)#每个图片多次加噪声，得到多张图片，将同级特征先按通道合并。
                        f_B = torch.cat((f_B, feats_B[i][lvl]), dim=1)
                    diff = torch.abs(layer(f_A)  - layer(f_B) )
                    if lvl<self.layers_perturb and perturbation:
                        diff = perturbation(diff, o_l)
                    if lvl!=0:
                        diff = diff + x
                    lvl+=1
                    
                else:
                    diff = layer(diff)# 改变通道数，进入下一个stage，统一通道数
                    x = F.interpolate(diff, scale_factor=2, mode="bilinear")# 上采样，统一分辨率
            del diff, f_A, f_B
            # Classifier
            cm = self.clfr_stg2(self.relu(self.clfr_stg1(x)))
            if cm.shape[2] != self.img_size:
                cm = F.interpolate(cm, size=(self.img_size,self.img_size), mode='bilinear', align_corners=True)
            return cm