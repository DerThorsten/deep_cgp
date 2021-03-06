#!/usr/bin/env python
from __future__ import print_function

import math
# from itertools import count
# import torch
# import torch.autograd
# import torch.nn.functional as F
# from torch.autograd import Variable
from torch import nn
import torch
import torch.nn.functional as F

# from http://code.activestate.com/recipes/577514-chek-if-a-number-is-a-power-of-two/
# Author: A.Polino
def is_power2(num):
    'states if a number is a power of two'
    return num != 0 and ((num & (num - 1)) == 0)





class Classifier(nn.Module):

    def __init__(self, x_j_channels_in, x_b_channels_in):
        super(Classifier, self).__init__()


        # numbers 
        self.n_classes = 5
        self.x_j_channels_in = x_j_channels_in
        self.x_b_channels_in = x_b_channels_in

        self.n_edge_feat = 512


        # modules
        self.edge_module = nn.Sequential(
            nn.Linear(x_b_channels_in, self.n_edge_feat),
            nn.PReLU(),
            nn.BatchNorm1d(self.n_edge_feat),
            nn.Dropout(0.1),
            #
            nn.Linear(self.n_edge_feat, self.n_edge_feat),
            nn.PReLU(),
            nn.BatchNorm1d(self.n_edge_feat),
            nn.Dropout(0.1),
            #
            nn.Linear(self.n_edge_feat, self.n_edge_feat),
            nn.PReLU(),
            nn.BatchNorm1d(self.n_edge_feat),
            nn.Dropout(0.1)
        )

        # j module
        classifier_n_in  = 3*self.n_edge_feat  + x_j_channels_in

        self.classifier = nn.Sequential(
            nn.Linear(classifier_n_in, 2048),
            nn.PReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.PReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.1),
            nn.Linear(2048, self.n_classes),
            nn.Softmax()
        )


    def forward(self, x_j, x_b0, x_b1, x_b2):
        
        # improve the boundary features
        x_b0   = self.edge_module(x_b0)
        x_b1   = self.edge_module(x_b1)
        x_b2   = self.edge_module(x_b2)

        # concat
        x_j = torch.cat([x_j, x_b0, x_b1, x_b2], 1)


        return self.classifier(x_j)







# => TODO
# run this for each
# each on its own!!
# and have an own mlp for each
# edge! !!!!!!
class SimpleMaskFeatureExtractor(nn.Module):

    def __init__(self, channels_in):
        super(SimpleMaskFeatureExtractor, self).__init__()

        self.channels_in = channels_in
        self.channels_out = 2*self.channels_in
        #self.batch_norm = nn.BatchNorm1d(self.channels_out)
        #self.nonlin = nn.PReLU()

    def forward(self, x, mask_j):


        batch_size = int(mask_j.size(0))
        shape_x0 = int(mask_j.size(2))
        shape_x1 = int(mask_j.size(2))
        n_pixel =  shape_x0 * shape_x1

        mask_j0 = mask_j[:, 0, :, :].contiguous()
        mask_j1 = mask_j[:, 1, :, :].contiguous()
        mask_j2 = mask_j[:, 2, :, :].contiguous()

        # flatten

        x_flat = x.view(batch_size, -1, n_pixel)

        flat_mask0 = mask_j0.view(batch_size,1, n_pixel)
        flat_mask1 = mask_j1.view(batch_size,1, n_pixel)
        flat_mask2 = mask_j2.view(batch_size,1, n_pixel)


      
        # get size of boundaries
        size_j0 = flat_mask0.sum(2).float()[:,:,0]
        size_j1 = flat_mask1.sum(2).float()[:,:,0]
        size_j2 = flat_mask2.sum(2).float()[:,:,0]

      
        

        #print(x_flat.data.numpy().shape, flat_mask0.data.numpy().shape)

        mult_res0 = x_flat * flat_mask0.expand_as(x_flat)
        mult_res1 = x_flat * flat_mask1.expand_as(x_flat)
        mult_res2 = x_flat * flat_mask2.expand_as(x_flat)



        x0_sum = (mult_res0).sum(2)[:,:,0]
        x1_sum = (mult_res1).sum(2)[:,:,0]
        x2_sum = (mult_res2).sum(2)[:,:,0]

        x0_mean = x0_sum /  size_j0.expand_as(x0_sum)
        x1_mean = x1_sum /  size_j1.expand_as(x1_sum)
        x2_mean = x2_sum /  size_j2.expand_as(x2_sum)


        # cannot use min since zerors from mask
        #print("aa", (mult_res0).max(2))
        x0_max,_ = (mult_res0).max(2)
        x1_max,_ = (mult_res1).max(2)
        x2_max,_ = (mult_res2).max(2)

        x0_max = x0_max[:,:,0]
        x1_max = x1_max[:,:,0]
        x2_max = x2_max[:,:,0]



        xb_0 = torch.cat([x0_sum, x0_mean, x0_max],1)
        xb_1 = torch.cat([x1_sum, x1_mean, x1_max],1)
        xb_2 = torch.cat([x2_sum, x2_mean, x2_max],1)


        return xb_0, xb_1, xb_2


class ReImageFlatFeatures(nn.Module):

    def __init__(self, spatial_shape):
        super(ReImageFlatFeatures, self).__init__()
        self.shaptial_shape = spatial_shape


    def forward(self, x):
        shaptial_shape = self.shaptial_shape
        batch_size = int(x.size(0))

        x_unflat = x.view(batch_size, -1, 1, 1)
        x_unflat_image = x.expand((batch_size,-1,shaptial_shape[0], shaptial_shape[1]))

        return x_unflat_image






class InitModule(nn.Module):

    def __init__(self, channels_in, channels_out):
        super(InitModule, self).__init__()


        self.conv_0 = torch.nn.Conv2d(in_channels=channels_in,
            out_channels=channels_out,kernel_size=(3,3),padding=(1,1))
        self.nonlin = torch.nn.PReLU()

    def forward(self, x):
        x = self.conv_0(x)
        x = self.nonlin(x)
        return x


class ConvModule(nn.Module):

    def __init__(self, channels_in, channels_out):
        super(ConvModule, self).__init__()

        self.batch_norm = torch.nn.BatchNorm2d(num_features=channels_in)
        self.nonlin = torch.nn.PReLU()
        self.dropout    = torch.nn.Dropout2d(p=0.5)
        self.conv_0 = torch.nn.Conv2d(in_channels=channels_in,
            out_channels=channels_out,kernel_size=(3,3),padding=(1,1))
               
       

    def forward(self, x):

        x = self.batch_norm(x)
        x = self.nonlin(x)
        x = self.dropout(x)
        x = self.conv_0(x)
        
        return x


class ConvModuleEnd(nn.Module):

    def __init__(self, channels_in, channels_out):
        super(ConvModuleEnd, self).__init__()

        self.batch_norm_in = torch.nn.BatchNorm2d(num_features=channels_in)
        self.nonlin_in = torch.nn.PReLU()
        self.dropout    = torch.nn.Dropout2d(p=0.5)
        self.conv_0 = torch.nn.Conv2d(in_channels=channels_in,
            out_channels=channels_out,kernel_size=(3,3),padding=(1,1))
        self.batch_norm_out = torch.nn.BatchNorm2d(num_features=channels_out)
        self.nonlin_out = torch.nn.PReLU()
       

    def forward(self, x):

        x = self.batch_norm_in(x)
        x = self.nonlin_in(x)
        x = self.dropout(x)
        x = self.conv_0(x)
        x = self.batch_norm_out(x)
        x = self.nonlin_out(x)

        return x


class DenseBlock(nn.Module):

    def __init__(self, channels_in, channels_out, length=5):
        super(DenseBlock, self).__init__()

        self.length = length


        self.conv_blocks = [None]*self.length
        for i in range(self.length):
            current_channels_in = channels_in * (i+1)
            if i + 1 < self.length:
                self.conv_blocks[i] = ConvModule(channels_in=current_channels_in, channels_out=channels_out)
            else:
                self.conv_blocks[i] = ConvModuleEnd(channels_in=current_channels_in, channels_out=channels_out)

        self.conv_blocks = nn.ModuleList(self.conv_blocks)
    def forward(self, x0):

        current_input_list = [x0]

        for i in range(self.length):
            current_input_tensor = torch.cat(current_input_list, 1)
            current_output =  self.conv_blocks[i](current_input_tensor)
            current_input_list.append(current_output)
        
        return current_output


class J3Module(nn.Module):


    def __init__(self, channels_in, patch_size, fully_connected_size):
        super(J3Module, self).__init__()

        # check input
        if(not is_power2(patch_size)):
            raise RuntimeError('patch_size must be a power of 2, %d is not' % patch_size)
        if(not is_power2(fully_connected_size)):
            raise RuntimeError('fully_connected_size must be a power of 2, %d is not' % fully_connected_size)

        self.channels_in = channels_in
        self.patch_size = patch_size
        self.fully_connected_size = fully_connected_size

        p_in   = int(math.log2(patch_size))
        p_stop = int(math.log2(fully_connected_size))



        # how often do we need to pool
        self.n_downsample_steps = p_in - p_stop

        #print("n_downsample_steps", self.n_downsample_steps)

        # the init module 
        channels_out = 10
        self.init_module = InitModule(channels_in=channels_in, channels_out=channels_out)

        # pooling
        self.pool = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        # conv 
        self.conv_layers = []
        for i in range(self.n_downsample_steps):
            conv_layer = DenseBlock(channels_in=channels_out, channels_out=channels_out)
            self.conv_layers.append(conv_layer)

        channels_out = 10
        #classifier_n_in = (fully_connected_size**2) * channels_out
        
        # mask feature layer
        self.simple_mask_feature =  SimpleMaskFeatureExtractor(channels_in=channels_out)


        # at the end we have
        x_j_channels_in = (self.n_downsample_steps + 1) * channels_out * (fully_connected_size**2)

        x_b_channels_in = (self.n_downsample_steps) * channels_out * 3

        # the final classifier
        self.classifier = Classifier(x_j_channels_in=x_j_channels_in,
                                     x_b_channels_in=x_b_channels_in)

    def crop(self, x, in_size, out_size):

        # lets say in_size = 128
        # and out size 16
        # start = 128 - 16 / 2
        s = (in_size - out_size)//2
        return x[:,:,s:s+out_size, s:s+out_size]

    def forward(self,  x):
        #print("init")
        #print(x.size())


        mask_j = x[:,1:4,:,:]

        current_res_mask_j = mask_j 

        fully_connected_size_crops = []



        x = self.init_module(x)
        current_size = self.patch_size


        xb_0 = []
        xb_1 = []
        xb_2 = []

        for i in range(self.n_downsample_steps):
            #print("DS",i)
            conv_layer = self.conv_layers[i]


            x = conv_layer(x)



            ff_0, ff_1, ff_2 = self.simple_mask_feature(x, current_res_mask_j)
            xb_0.append(ff_0)
            xb_1.append(ff_1)
            xb_2.append(ff_2)

            x_crop_fully = self.crop(x, in_size=current_size, out_size=self.fully_connected_size)
            fully_connected_size_crops.append(x_crop_fully)


            # pool features
            x = self.pool(x)

            # pool masks
            current_res_mask_j = self.pool(current_res_mask_j)

            # update current size
            current_size = current_size//2

        fully_connected_size_crops.append(x)

        x_j = torch.cat(fully_connected_size_crops,1)



        # convert the image to flat 
        # features
        x_j = x_j.contiguous()
        x_j = x_j.view(x_j.size(0), -1)

        # bounary features
        xb_0 = torch.cat(xb_0,1)
        xb_1 = torch.cat(xb_1,1)
        xb_2 = torch.cat(xb_2,1)



        #print('shape',x.size())
        res = self.classifier(x_j, xb_0, xb_1, xb_2)

        return res