1

import numpy
import numbers
import nifty.segmentation
from .hl_cgp import HlCgp
from .j3_module import J3Module

import torch
import torch.nn

from torch.autograd import Variable



class Cell0PatchExtrator(object):
    def __init__(self, hl_cgp, node_gt, image, radius):

        self.hl_cgp = hl_cgp
        self.j3_labels = self.hl_cgp.j3_labels
        self.node_gt = node_gt
        self.image = image[...,None] if image.ndim == 2 else image
        self.cell_1_labels = self.hl_cgp.cell_1_labels
        self.radius = radius
        self.shape = self.image.shape[0:2]

        self.cell_0_geometry = self.hl_cgp.geometry[0]
        self.cell_1_geometry = self.hl_cgp.geometry[1]
        self.cell_0_bounds = self.hl_cgp.cell_bounds[0]

        # get cell 1 gt 
        if node_gt is not None:
            self.cell_1_gt = self.hl_cgp.cell_1_gt(self.node_gt)
        else:
            self.cell_1_gt = None
       


        # padding
        padding = ((self.radius, self.radius), (self.radius, self.radius), (0,0))
 
    
        # pad raw data
        self.padded_image = numpy.pad(self.image, pad_width=padding, mode='reflect')
        ones = numpy.ones(self.shape, dtype='bool')
        self.padding_indicator = numpy.pad(ones, pad_width=padding[0:2], mode='constant',constant_values=0)
      
    def __len__(self):
        return self.hl_cgp.n_cells[0]

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

        cell_1_gt = [None]*n_bounds

        for i in range(n_bounds):
            cell_1_label = bounds[i]
            cell_1_index = cell_1_label - 1

            if self.cell_1_gt is not None:
                cell_1_gt[i] = self.cell_1_gt[cell_1_index]

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

            patch[coords_x0,coords_x1] = 1
            # image_patch[coords_x0,coords_x1,:] = 0


        #print(p_begin, p_end, image_patch.shapeape)
        images = [
            image_patch, 
            cell_1_labels_patch, 
            padding_indicator_patch[:,:,None]
        ]
        images = numpy.concatenate(images, axis=2)


        if self.cell_1_gt is not None:
            cell_1_gt = numpy.array(cell_1_gt)
            int_gt  =  numpy.round(cell_1_gt)
            gt_quali = 1.0 - numpy.sum(numpy.abs(int_gt-cell_1_gt))/3.0
            int_gt = [int(g) for g in int_gt]

            scalar_gt = None
            if int_gt == [0,0,0]:
                scalar_gt = 0
            elif int_gt == [0,1,1]:
                scalar_gt = 1
            elif int_gt == [1,0,1]:
                scalar_gt = 2
            elif int_gt == [1,1,0]:
                scalar_gt = 3
            elif int_gt == [1,1,1]:
                scalar_gt = 4
            else:
                raise RuntimeError("mph...")

            return images, scalar_gt, gt_quali

        else:
            return images

from cachetools import RRCache
import random

class IsbiJ3Feeder(object):
    
    def __init__(self, raw_slices, pmap_slices, gt_slices, patch_radius, batch_size, cache_size=5,
        p_from_cache=0.97):

        self.raw_slices = raw_slices
        self.pmap_slices = pmap_slices

        self.gt_slices  = gt_slices

        self.patch_radius = patch_radius

        self.batch_size = batch_size

        self.n_slices = raw_slices.shape[0]
        

        self.cache = RRCache(maxsize=cache_size)
        self.p_from_cache = p_from_cache 
    
    def __call__(self):
        
        batch_images           = [None]*self.batch_size
        batch_gt               = [None]*self.batch_size
        batch_gt_quali  = [None]*self.batch_size

        for i in range(self.batch_size):

            # get a patch extractor
            patch_extractor = self.__get_random_slice_data()
            

            # get a random 0-cell index
            # but we only consider 0-cells with
            # a since of 3
            n_patches = len(patch_extractor)
            

            j3_labels = patch_extractor.j3_labels
            assert len(j3_labels) >=1

            done = False
            while(not done):
                rand_index = random.randint(0, len(j3_labels)-1)
                cell_0_label = j3_labels[rand_index]
                assert cell_0_label >= 1
                cell_0_index = cell_0_label - 1
                try:
                    done = True
                    img, gt, gt_quali = patch_extractor[cell_0_index]    
                except:
                    print("hubs....")
                    done  = False

            # img shape atm : x,y,c
            # => desired 1,c,x,y
            img = numpy.rollaxis(img, 2,0)[None,...]
            batch_images[i] = img

          
            batch_gt[i] = gt
            batch_gt_quali[i] = gt_quali


        batch_images = numpy.concatenate(batch_images,axis=0)
        #print("batch_gt",batch_gt)
        batch_gt = numpy.array(batch_gt)
        batch_gt_quali = numpy.array(batch_gt_quali)
        # batch_images: (batch_size, c, x,y)
        # batch_gt:     (batch_size, 3)
        return batch_images, batch_gt, batch_gt_quali
    
    def __get_random_slice_data(self):

        take_from_cache = random.random() >= (1.0 - self.p_from_cache )
        if take_from_cache and len(self.cache)>0:
            # get random item from cache via pop 
            # (since this is a random cache this 
            # will lead to a random item)
            per_slice_data = self.__get_random_from_cache()
            return per_slice_data

        else:
            # (maybe) compute new
            # random slice
            slice_index = random.randint(0, self.n_slices-1)

            # get the per_slice_data from cache iff already in cache.
            # Iff not in cache, compute per_slice_data and put to cache
            per_slice_data = self.__force_to_cache(slice_index=slice_index)
            return per_slice_data
    
    def __get_random_from_cache(self):
        assert len(self.cache) > 0
        slice_index, per_slice_data = self.cache.popitem()
        self.cache[slice_index] = per_slice_data
        return per_slice_data

    def __force_to_cache(self, slice_index):
        if slice_index in self.cache:
            per_slice_data = self.cache[slice_index]
            return per_slice_data
        else:
            per_slice_data = self.__compute_per_slice_data(slice_index)
            self.cache[slice_index] = per_slice_data
            return per_slice_data

    def __edge_gt_to_node_gt(self, edge_gt):
        # the edge_gt is on membrane level
        # 0 at membranes pixels
        # 1 at non-membrane piper_slice_dataxels

        seeds = nifty.segmentation.localMaximaSeeds(edge_gt)
        growMap = nifty.filters.gaussianSmoothing(1.0-edge_gt, 1.0)
        growMap += 0.1*nifty.filters.gaussianSmoothing(1.0-edge_gt, 6.0)
        gt = nifty.segmentation.seededWatersheds(growMap, seeds=seeds)

        return gt

    def __compute_per_slice_data(self, slice_index):

        raw_slice   = self.raw_slices[slice_index,:,:]
        gt_slice    = self.gt_slices[slice_index,:,:]
        pmap_slice  = self.pmap_slices[slice_index,:,:]
        edge_gt     = self.gt_slices[slice_index,:,:]
        node_gt     = self.__edge_gt_to_node_gt(edge_gt)

        # randomized overseg 
        threshold = random.uniform(0.275, 0.55)
        overseg = nifty.segmentation.distanceTransformWatersheds(pmap_slice.copy(), 
            threshold=threshold)
        hl_cgp = HlCgp(overseg)
        cell_0_patch_extrator = Cell0PatchExtrator(hl_cgp, image=raw_slice, 
            node_gt=node_gt,
            radius=self.patch_radius)

        return cell_0_patch_extrator


class IsbiJ3(object):

    def __init__(self, patch_radius, fully_connected_size):
        
        self.patch_radius = patch_radius

        # input channels for nn
        self.channels_in = 5
        # input size for nn
        self.patch_size = self.patch_radius*2 + 2

        self.fully_connected_size = fully_connected_size

        self.nn_j3 = J3Module(channels_in=self.channels_in, 
            patch_size=self.patch_size,
            fully_connected_size=self.fully_connected_size)

        self.__loss_function = torch.nn.CrossEntropyLoss()

    def loss_function(self):
        return self.__loss_function 

    def model(self):
        return self.nn_j3



    def feeder(self, raw_slices, pmap_slices, gt_slices, batch_size=20, cache_size=20):
        """ spits out random training examples and will also compute the superpixels
        to train the net
        """
        return IsbiJ3Feeder(raw_slices=raw_slices, pmap_slices=pmap_slices, gt_slices=gt_slices,
            patch_radius=self.patch_radius, batch_size=batch_size, cache_size=cache_size)


    def predictor(self, hl_cgp, raw_slice):
        """to prediction all junctions of a **single** image
        """
        
        cell_0_extractor = Cell0PatchExtrator(hl_cgp=hl_cgp, node_gt=None, image=raw_slice, radius=self.patch_radius)



        class Predictor(object):
            def __init__(self, extractor, nn):
                self.extractor = extractor
                self.nn = nn


            def predict(self, cell_0_index):
                nn_input_image = self.extractor[cell_0_index].astype('float32')


                # nn_input_image shape atm : x,y,c
                # => desired 1,c,x,y
                nn_input_image = numpy.rollaxis(nn_input_image, 2,0)[None,...]
                nn_input_image = Variable(torch.from_numpy(nn_input_image))
                res = self.nn(nn_input_image)
                

                #def softmax(x):
                #    """Compute softmax values for each sets of scores in x."""
                #    return numpy.exp(x) / numpy.sum(numpy.exp(x))


                try:
                    res_numpy = res.data.numpy()
                except:
                    res_numpy = res.numpy()


                # TODO convert this
                # to usable gt
                return res_numpy
                #return softmax(res_numpy)

        return Predictor(extractor=cell_0_extractor, nn=self.nn_j3)


if __name__ == "__main__":

    pass