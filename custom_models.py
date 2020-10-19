import layer_utils as lu
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


class UpConv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
    ):
        super(UpConv,self).__init__()
        self.deconv = nn.ConvTranspose2d( in_channels, out_channels, kernel_size=2,stride=2, dilation=1, padding=0, bias=False)
    
    def forward(self, x):
        x = self.deconv(x)
        return x
    
class Conv2dSecuences(nn.Module):
    def __init__(
            self,
            sequence_size,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True
    ):
        super(Conv2dSecuences,self).__init__()
        self.convs = nn.Sequential(*[
            lu.Conv2dReLU(2**(i+1)*512,2**(i)*512, kernel_size=kernel_size, padding=padding) for i in reversed(range(int.bit_length(sequence_size)-1))
        ])

    def forward(self, x):
        x = x.contiguous().view(x.shape[0],-1,*(x.size()[3:]))
        
        x = self.convs(x)
        return x


class ReplicatorModel(nn.Module):
    def __init__(self, params_model):
        super(ReplicatorModel, self).__init__()
        self.input_dim = params_model["input_dim"]
        self.sequence_size = params_model["sequence_size"]
        
        uNetModel = smp.Unet('vgg16_bn', encoder_weights='imagenet')
        for param in uNetModel.parameters():
            param.requires_grad = False
        self.encoder = uNetModel.encoder
        
        self.centerConv1 = lu.Conv2dReLU(512,1024,kernel_size=3, padding=1)
        #concat with LSTM output
        self.centerSequenceConv1 = Conv2dSecuences(self.sequence_size, kernel_size=1)
        self.centerSequenceConv2 = lu.Conv2dReLU(512,1024,kernel_size=3, padding=1)
        
        #self.centerLSTM = nn.LSTM(512*(self.input_dim//32)**2,256*(self.input_dim//32)**2,lstm_num_layers,bidirectional=True)
        self.centerConv2 = lu.Conv2dReLU(2048,1024,kernel_size=1)
        self.centerConv3 = lu.Conv2dReLU(1024,1024,kernel_size=3, padding=1)
        
        self.up1 = UpConv(1024, 512, kernel_size=3, padding=1)
        #concat with encoder output
        self.upConv1a = lu.Conv2dReLU(1024,512, kernel_size=3,padding=1)
        self.upConv1b = lu.Conv2dReLU(512,512, kernel_size=3,padding=1)
        
        self.up2 = UpConv(512, 256, kernel_size=3, padding=1)
        #concat with encoder output
        self.upConv2a = lu.Conv2dReLU(768,256, kernel_size=3,padding=1)
        self.upConv2b = lu.Conv2dReLU(256,256, kernel_size=3,padding=1)
        
        self.up3 = UpConv(256, 128, kernel_size=3, padding=1)
        #concat with encoder output
        self.upConv3a = lu.Conv2dReLU(384,128, kernel_size=3,padding=1)
        self.upConv3b = lu.Conv2dReLU(128,128, kernel_size=3,padding=1)
        
        self.up4 = UpConv(128, 64, kernel_size=3, padding=1)
        #concat with encoder output
        self.upConv4a = lu.Conv2dReLU(192,64, kernel_size=3,padding=1)
        self.upConv4b = lu.Conv2dReLU(64,64, kernel_size=3,padding=1)
        
        self.up5 = UpConv(64, 32, kernel_size=3, padding=1)
        #concat with encoder output
        self.upConv5a = lu.Conv2dReLU(96,32, kernel_size=3,padding=1)
        self.upConv5b = lu.Conv2dReLU(32,32, kernel_size=3,padding=1)
        self.upConv5c = nn.Conv2d(32,3, kernel_size=1,padding=0)#output
        
        
    def forward(self, x):
        targetInput = x[:,0,:,:,:]
        secuenceInput = x[:,1:,:,:,:]
        
        e5,e4,e3,e2,e1,x = self.encoder(targetInput)
        secuenceIF = secuenceInput.reshape(-1, *(secuenceInput.size()[2:]))

        secuenceIF = self.encoder(secuenceInput.reshape(-1, *(secuenceInput.size()[2:])))[-1]
        secuenceIF = secuenceIF.view(*(secuenceInput.size()[:2]),*(x.size()[1:]))
        secuenceIF = self.centerSequenceConv1(secuenceIF)
        secuenceIF = self.centerSequenceConv2(secuenceIF)
        x = self.centerConv1(x)
        x = torch.cat((x,secuenceIF),dim=1)
        x = self.centerConv2(x)
        x = self.centerConv3(x)
        
        x = self.up1(x)
        x = torch.cat((e1,x),dim=1)
        x = self.upConv1a(x)
        x = self.upConv1b(x)
        
        x = self.up2(x)
        x = torch.cat((e2,x),dim=1)
        x = self.upConv2a(x)
        x = self.upConv2b(x)
        
        x = self.up3(x)
        x = torch.cat((e3,x),dim=1)
        x = self.upConv3a(x)
        x = self.upConv3b(x)
        
        x = self.up4(x)
        x = torch.cat((e4,x),dim=1)
        x = self.upConv4a(x)
        x = self.upConv4b(x)
        
        x = self.up5(x)
        x = torch.cat((e5,x),dim=1)
        x = self.upConv5a(x)
        x = self.upConv5b(x)
        
        x = self.upConv5c(x)
        
        return x
        