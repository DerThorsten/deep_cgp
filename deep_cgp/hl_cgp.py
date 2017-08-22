

import nifty.cgp as ncgp
import nifty.graph as ngraph
import nifty.ground_truth as ngt
import numpy



class HlCgp(object):

    def __init__(self, labels):
        assert labels.min() == 1


        self.labels = labels
        self.cgp = ncgp.TopologicalGrid2D(labels)
        self.n_cells = self.cgp.numberOfCells
        self.cell_bounds = self.cgp.extractCellsBounds()
        self.geometry = self.cgp.extractCellsGeometry()
        self.filled_tgrid = ncgp.FilledTopologicalGrid2D(self.cgp)

        # all junction of size 3
        j3_labels = []
        bounds0 = self.cell_bounds[0]
        for cell_0_index in range(self.n_cells[0]):
            if len(bounds0[cell_0_index]) == 3:
                cell_0_label = cell_0_index + 1
                j3_labels.append(cell_0_label)
        self.j3_labels  = j3_labels               
        # build regular graph
        self.graph = ngraph.UndirectedGraph(self.n_cells[2])
        uv_ids_cgp = numpy.array(self.cell_bounds[1])-1
        self.graph.insertEdges(uv_ids_cgp)
        self.cgp_edge_to_graph_edge = self.graph.findEdges(uv_ids_cgp)


        self.cell_1_labels = self.cell_label_image(cell_type=1)

        self.bounded_by = {
            1:self.cell_bounds[0].reverseMapping(),
            2:self.cell_bounds[1].reverseMapping()
        }


    def cell_label_image(self, cell_type):
        c = [False,False,False]
        c[cell_type] = True
        img = self.filled_tgrid.cellMask(showCells=c)
        img[img!=0] -= self.filled_tgrid.cellTypeOffset[1]

        return img


    def cell_1_gt(self, pixel_wise_gt):
        # map the gt to the edges
        overlap = ngt.overlap(segmentation=self.labels,
                                       groundTruth=pixel_wise_gt)
        cell_1_gt = overlap.differentOverlaps(numpy.array(self.cell_bounds[1]))
        #print(cell_1_gt)
        return cell_1_gt




    # 0 0 0  <=>  0    <=>     0
    # 0 0 1  <=>  1     
    # 0 1 0  <=>  2     
    # 0 1 1  <=>  3    <=>     1
    # 1 0 0  <=>  4     
    # 1 0 1  <=>  5    <=>     2
    # 1 1 0  <=>  6    <=>     3
    # 1 1 1  <=>  7    <=>     4
    def junction_vector_to_scalar_gt(j3_gt):
        relabel = numpy.array([0,-1,-1, 1,-1, 2, 3, 4])
        sclar_with_gaps =  j3_gt[:,0]*4 + j3_gt[:,1]*2 + j3_gt[:,2]
        s = relabel[sclar_with_gaps]

        assert s.min() >=0
        assert s.max() <=4
        return s

