"""
2D oriented Non-Max-Suppression

thanks https://github.com/MhLiao/TextBoxes_plusplus/blob/master/examples/text/nms.py
"""
import numpy as np
from shapely.geometry import Polygon, MultiPoint

def polygon_iou(pts1, pts2):
    """
    Intersection over union between two shapely polygons.
    """
    poly1 = Polygon(pts1).convex_hull
    poly2 = Polygon(pts2).convex_hull
    union_poly = np.concatenate((pts1,pts2))
    if not poly1.intersects(poly2): # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            #union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def nms(boxes, scores, iou_thresh, max_output_size):
    '''
    Input:
        boxes: (N,4,2) [x,y]
        scores: (N)
    Return:
        nms_mask: (N)
    '''
    indices = sorted(range(len(scores)), key=lambda k: -scores[k])
    box_num = len(boxes)
    nms_mask = [True]*box_num
    for i in range(box_num):
        ii = indices[i]
        if not nms_mask[ii]:
            continue
        print("oriented_NMS-o-loop: %d" % i)
        for j in range(box_num):
            jj = indices[j]
            if j == i:
                continue
            if not nms_mask[jj]:
                continue
            box1 = boxes[ii]
            box2 = boxes[jj] 
            box1_score = scores[ii]
            box2_score = scores[jj] 
            #box_i = [box1[0],box1[1],box1[4],box1[5]]
            #box_j = [box2[0],box2[1],box2[4],box2[5]]
            poly1 = Polygon(box1).convex_hull
            poly2 = Polygon(box2).convex_hull
            iou = polygon_iou(box1,box2)
     
            if iou > iou_thresh:
                if box1_score > box2_score:
                    nms_mask[jj] = False  
                if box1_score == box2_score and poly1.area > poly2.area:
                    nms_mask[jj] = False  
                if box1_score == box2_score and poly1.area<=poly2.area:
                    nms_mask[ii] = False  
                    break
            '''
            if abs((box_i[3]-box_i[1])-(box_j[3]-box_j[1]))<((box_i[3]-box_i[1])+(box_j[3]-box_j[1]))/2:
                if abs(box_i[3]-box_j[3])+abs(box_i[1]-box_j[1])<(max(box_i[3],box_j[3])-min(box_i[1],box_j[1]))/3:
                    if box_i[0]<=box_j[0] and (box_i[2]+min(box_i[3]-box_i[1],box_j[3]-box_j[1])>=box_j[2]):
                        nms_mask[jj] = False
            '''
    trim = np.sum(nms_mask) - max_output_size
    if trim > 0:
        for i in range(box_num):
            ii = indices[box_num-i-1]
            if nms_mask[ii]:
                nms_mask[ii] = False
                trim -= 1
                if trim <= 0:
                    break
        
    return np.asarray(nms_mask, dtype=np.bool)

