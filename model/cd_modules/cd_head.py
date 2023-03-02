# Change detection head

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
from model.cd_modules.psp import _PSPModule
from .cd_head_v2 import Block, AttentionBlock
def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    '''
    Get the number of input layers to the change detection head.
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
    

class cd_head(nn.Module):
    '''
    Change detection head.
    '''

    def __init__(self, feat_scales, out_channels=2, inner_channel=None, channel_multiplier=None, img_size=256, psp=False):
        super(cd_head, self).__init__()

        # Define the parameters of the change detection head
        self.feat_scales    = feat_scales
        self.in_channels    = get_in_channels(feat_scales, inner_channel, channel_multiplier)
        self.img_size       = img_size

        # Convolutional layers before parsing to difference head
        self.diff_layers = nn.ModuleList()
        for feat in feat_scales:
            if psp:
                self.diff_layers.append(_PSPModule(in_channels=get_in_channels([feat], inner_channel, channel_multiplier),
                                        bin_sizes=[1, 2, 3, 6]))
            else:
                self.diff_layers.append(nn.Conv2d(  in_channels=get_in_channels([feat], inner_channel, channel_multiplier), 
                                                    out_channels=get_in_channels([feat], inner_channel, channel_multiplier), 
                                                    kernel_size=3, 
                                                    padding=1))

        #MLP layer to reduce the feature dimention
        if psp:
            self.in_channels = int(self.in_channels/4)
        
        self.conv1_final = nn.Conv2d(self.in_channels, 64, kernel_size=1, padding=0)

        #Get final change map
        self.conv2_final = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, feats_A, feats_B):

        feats_diff = []
        c=0
        for layer in self.diff_layers:
            x = layer(torch.abs(feats_A[self.feat_scales[c]] - feats_B[self.feat_scales[c]]))
            #torch.abs(layer(feats_A[self.feat_scales[c]]) - layer(feats_B[self.feat_scales[c]]))
            feats_diff.append(x)
            c+=1
        
        c=0
        for i in range(0, len(feats_diff)):
            if feats_diff[i].size(2) != self.img_size:
                feat_diff_up = F.interpolate(feats_diff[i], size=(self.img_size, self.img_size), mode="bilinear")
            else:
                feat_diff_up = feats_diff[i]
            
            #Concatenating upsampled features to ''feats_diff_up''
            if c==0:
                feats_diff_up = feat_diff_up
                c+=1
            else:
                feats_diff_up = torch.cat((feats_diff_up, feat_diff_up), dim=1)

        cm = self.conv2_final(self.relu(self.conv1_final(feats_diff_up)))

        return cm

class cd_head_v2(nn.Module):
    '''
    Change detection head (version 2).
    '''
    # feat_scales 表示要使用哪层特征用于变化检测，因为Unet返回的编码特征有15层，解码特征有20层，因此最大为14。
    # feat_scales=[2, 5, 8, 11, 14], inner_channel=128, channel_multiplier=[1,2,4,8,8],time_steps=[50,100,400]
    def __init__(self, feat_scales, out_channels=2, inner_channel=None, channel_multiplier=None, img_size=256, time_steps=None):
        super(cd_head_v2, self).__init__()

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
