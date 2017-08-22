# -*- coding: utf-8 -*-

import sys
import random

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
fully_connected_size = 16
batch_size = 100

name = "very_legendary_14"
filename_model = "/home/tbeier/src/deep_cgp/sandbox/%s_model_fb.pytorch"%name
filename_opt = "/home/tbeier/src/deep_cgp/sandbox/%s_optimizer_fb.pytorch"%name


def adjust_learning_rate(optimizer, epoch, base_lr):
    lr = base_lr * (0.80 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(lr,0.0000001)
    return lr


if True:


    # all the stuff for isbi
    isbi_j3 = deep_cgp.IsbiJ3(patch_radius=patch_radius, fully_connected_size=fully_connected_size)

    feeder = isbi_j3.feeder(raw_slices=raw_dset,
        pmap_slices=pmap_dset, gt_slices=gt_dset, 
        batch_size=batch_size)




    loss_function = isbi_j3.loss_function()
    model = isbi_j3.model()


    # the optimizer
    learning_rate = 0.025
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)






    # load opt state
    try:
        if os.path.isfile(filename_opt):
            optimizer.load_state_dict(torch.load(filename_opt))
    except Exception as e:
        print("something went wrong during loading",str(e))

    # load parameters
    try:
        if os.path.isfile(filename_model):
            model.load_state_dict(torch.load(filename_model))
    except Exception as e:
        print("something went wrong during loading",str(e))


    model.train()
    # the actual optimization
    for epoch in range(10000):


        elr = adjust_learning_rate(optimizer, epoch+1, learning_rate)

        # get batch
        batch_imgs, batch_gt, batch_gt_quali = feeder()

        # ensure right dtype
        batch_imgs = numpy.require(batch_imgs, dtype='float32')
        batch_gt = numpy.require(batch_gt, dtype='int')
        batch_gt_quali = numpy.require(batch_gt_quali, dtype='float32')

        # convert numpy to torch create variables
        batch_imgs = Variable(torch.from_numpy(batch_imgs), requires_grad=False)
        batch_gt   = Variable(torch.from_numpy(batch_gt))
        batch_gt_quali   = Variable(torch.from_numpy(batch_gt_quali))


        # put trough net 
        x = model(batch_imgs)

        # get loss 
        loss = loss_function(x, batch_gt)

        # print loss
        print("epoch",epoch,"lr",elr,"loss", loss.data.numpy()[0])


        # save parameters
        torch.save(model.state_dict(), filename_model)

        # save optimizer
        torch.save(optimizer.state_dict(), filename_opt)

        # do gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


else:


    raw_dset  = raw_dsets['test']/255.0 - 0.7
    pmap_dset = pmap_dsets['test']
    #gt_dset   = gt_dsets['test']

    isbi_j3 = deep_cgp.IsbiJ3(patch_radius=patch_radius, fully_connected_size=fully_connected_size)


    slice_index = 10 

    threshold = 0.3
    overseg = nifty.segmentation.distanceTransformWatersheds(pmap_dset[slice_index, :,:].copy(), 
            threshold=threshold)

    hl_cgp = deep_cgp.HlCgp(overseg)


    predictor = isbi_j3.predictor(hl_cgp=hl_cgp, raw_slice=raw_dset[slice_index, :,:])

    print("load model")
    model = isbi_j3.model()
    model.load_state_dict(torch.load(filename_model))
    model.eval()
    cell_0_bounds = hl_cgp.cell_bounds[0]
    for cell_0_index in  range(hl_cgp.n_cells[0]):
        #print(cell_0_index,hl_cgp.n_cells[0])
        cell_0_label = cell_0_index + 1
        bounds = cell_0_bounds[cell_0_index]
        if len(bounds) == 3:

            prediction = predictor.predict(cell_0_index=cell_0_index)

            print(numpy.argmax(prediction))