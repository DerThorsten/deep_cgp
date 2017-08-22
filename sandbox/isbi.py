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



slice_index = 5

raw_dset  = raw_dsets['train']
pmap_dset = pmap_dsets['train']
gt_dset   = gt_dsets['train']


raw            = raw_dset[slice_index, :,:]
pmap           = pmap_dset[slice_index, :,:]
binary_edge_gt = gt_dset[slice_index, :,:]



shape = raw.shape
tshape = tuple([s*2 - 1 for s in shape])




# overseg 
overseg = nifty.segmentation.distanceTransformWatersheds(pmap, threshold=0.3)
hl_cgp = deep_cgp.HlCgp(overseg)


import numpy
import numbers

class Cell0PatchExtrator(object):
    def __init__(self, hl_cgp, image, radius):



        self.hl_cgp = hl_cgp
        self.image = image[...,None] if image.ndim == 2 else image
        self.cell_1_labels = self.hl_cgp.cell_1_labels
        self.radius = radius
        self.shape = self.image.shape[0:2]

        self.cell_0_geometry = self.hl_cgp.geometry[0]
        self.cell_1_geometry = self.hl_cgp.geometry[1]
        self.cell_0_bounds = self.hl_cgp.cell_bounds[0]

        # padding
        padding = ((self.radius, self.radius), (self.radius, self.radius), (0,0))
 
    
        # pad raw data
        self.padded_image = numpy.pad(self.image, pad_width=padding, mode='reflect')
        ones = numpy.ones(shape, dtype='bool')
        self.padding_indicator = numpy.pad(ones, pad_width=padding[0:2], mode='constant',constant_values=0)
      



    def __getitem__(self, cell_0_index):
        assert isinstance(cell_0_index, numbers.Integral)
        assert cell_0_index < self.hl_cgp.n_cells[0]

        cell_0_label = cell_0_index + 1

        # center coord of the junction
        # in topological coordinates
        tcoord = self.cell_0_geometry[cell_0_index][0]

        # => 4 coordinates in normal coordinates
        top_left     = tcoord[0]//2,   tcoord[1]//2
        bottom_right = tcoord[0]//2+1, tcoord[1]//2+1


        # switch to padded coordinates
        top_left = [c + self.radius for c in top_left]
        bottom_right = [c + self.radius for c in bottom_right]

        # patch begin and end
        p_begin = [c - self.radius     for c in top_left]
        p_end =   [c + self.radius + 1 for c in bottom_right]

        # get the patch of the image
        image_patch = self.padded_image[p_begin[0]: p_end[0], p_begin[1]: p_end[1], :].copy()
        padding_indicator_patch = self.padding_indicator[p_begin[0]: p_end[0], p_begin[1]: p_end[1]]

        bounds = self.cell_0_bounds[cell_0_index]
        n_bounds = len(bounds)

        shape = image_patch.shape[0:2]+(n_bounds,)
        cell_1_labels_patch = numpy.zeros(shape)


        for i in range(n_bounds):
            cell_1_label = bounds[i]
            cell_1_index = cell_1_label - 1
            cell_1_geo = numpy.array(self.cell_1_geometry[cell_1_index])
            c_coords = numpy.ceil(cell_1_geo/2).astype('uint32')
            f_coords = numpy.floor(cell_1_geo/2).astype('uint32')

            coords = numpy.concatenate([c_coords,f_coords],axis=0)

            # switch to junction coordinates and
            coords[:,0] -= p_begin[0]
            coords[:,1] -= p_begin[1]


            # switch to padded coordinates
            coords += self.radius

            patch = cell_1_labels_patch[...,i] 

            
            coords_x0 = coords[:,0]
            coords_x1 = coords[:,1]
            valid_mask = numpy.ones(coords_x0.shape,dtype='bool')
            valid_mask[coords_x0<0] = 0
            valid_mask[coords_x1<0] = 0
            valid_mask[coords_x0>=patch.shape[0]] = 0
            valid_mask[coords_x1>=patch.shape[1]] = 0

            coords_x0 = coords_x0[valid_mask]
            coords_x1 = coords_x1[valid_mask]

            patch[coords_x0,coords_x1] = 255
            image_patch[coords_x0,coords_x1,:] = 0


        #print(p_begin, p_end, image_patch.shapeape)
        return image_patch, cell_1_labels_patch,padding_indicator_patch

#print(raw.shape, raw.dtype)
traw = resize(raw,tshape, mode='reflect')
cell_0_patch_extrator = Cell0PatchExtrator(hl_cgp, image=raw, radius=(32-2)//2)


for cell_0_index in range(0,hl_cgp.n_cells[0],20):
    #print("cell_0_index",cell_0_index)


    image_patch, cell_1_labels_patch,padding_indicator_patch = cell_0_patch_extrator[cell_0_index]


    if 1:
        f = pylab.figure() 

        f.add_subplot(1, 3, 1)  
        pylab.imshow(image_patch[...,0],cmap='gray')

        f.add_subplot(1, 3, 2)  
        pylab.imshow(numpy.sum(cell_1_labels_patch,axis=2),cmap='jet')
        
        f.add_subplot(1, 3, 3)  
        pylab.imshow(padding_indicator_patch,cmap='jet')

        pylab.show()







if False:


    # create an instance
    instance = deep_cgp.Instance(labels=overseg)


    # create a training instance
    gt = deep_cgp.BinaryEdgeGt(binary_edge_gt)
    training_instance = deep_cgp.TrainingInstane(labels=overseg, gt=gt)


