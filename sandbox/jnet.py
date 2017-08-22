# -*- coding: utf-8 -*-

import sys
sys.path.append('/home/tbeier/src/deep_cgp/')
import deep_cgp

# to download data and unzip it
import os
import urllib.request
import zipfile

# to read tiff
import skimage.io

# to overseg
import nifty.segmentation
import numpy
import vigra
from skimage.transform import rescale, resize, downscale_local_mean



import pylab
import matplotlib.cm as cm



# torch
import torch
import torch.nn
from torch.autograd import Variable

#############################################################
# Download  ISBI 2012:
# =====================
# Download the  ISBI 2012 dataset 
# and precomputed results form :cite:`beier_17_multicut`
# and extract it in-place.
fname = "data.zip"
url = "http://files.ilastik.org/multicut/NaturePaperDataUpl.zip"
if not os.path.isfile(fname):
    urllib.request.urlretrieve(url, fname)
    zip = zipfile.ZipFile(fname)
    zip.extractall()


#############################################################
# Setup Datasets:
# =================
# load ISBI 2012 raw and probabilities
# for train and test set
# and the ground-truth for the train set
raw_dsets = {
    'train' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/raw_train.tif'),
    'test' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/raw_test.tif'),
}
# read pmaps and convert to 01 pmaps
pmap_dsets = {
    'train' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/probabilities_train.tif'),
    'test' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/probabilities_test.tif'),
}
pmap_dsets = {
    'train' : pmap_dsets['train'].astype('float32')/255.0,
    'test' : pmap_dsets['test'].astype('float32')/255.0
}
gt_dsets = {
    'train' : skimage.io.imread('NaturePaperDataUpl/ISBI2012/groundtruth.tif'),
    'test'  : None
}




raw_dset  = raw_dsets['train']/255.0 - 0.7
pmap_dset = pmap_dsets['train']
gt_dset   = gt_dsets['train']



# the patch extractor
patch_radius = (2**6-2)//2
patch_size = 2**6
print("patch size",patch_size)
batch_size = 30
isbi_feeder = deep_cgp.IsbiJ3Feeder(raw_slices=raw_dset, pmap_slices=pmap_dset,
    gt_slices=gt_dset, patch_radius=patch_radius, batch_size=batch_size)




# channels in => 5 

print("patch size",patch_size)
model = deep_cgp.J3Module(channels_in=5, patch_size=patch_size,
    fully_connected_size=16)


# the optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



loss_function = torch.nn.CrossEntropyLoss()

for x in range(100):


    batch_imgs, batch_gt, batch_gt_quali = isbi_feeder()

    batch_imgs = numpy.require(batch_imgs, dtype='float32')
    batch_gt = numpy.require(batch_gt, dtype='int')
    batch_gt_quali = numpy.require(batch_gt_quali, dtype='float32')

    # convert numpy to torch create variables
    batch_imgs = Variable(torch.from_numpy(batch_imgs), requires_grad=True)
    batch_gt   = Variable(torch.from_numpy(batch_gt))
    batch_gt_quali   = Variable(torch.from_numpy(batch_gt_quali))



    x = model(batch_imgs)
    loss = loss_function(x, batch_gt)

    print("loss", loss.data.numpy()[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
