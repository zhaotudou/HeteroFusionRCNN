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
        inter_area = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.area + poly2.area - inter_area
            #union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 0,0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
            inter_area = 0
    return iou, inter_area

def nms(boxes, scores, iou_thresh, max_output_size):
    '''
    Input:
        boxes: (N,4,2) [x,y]
        scores: (N)
    Return:
        nms_mask: (N)
    '''
    box_num = len(boxes)
    output_size = min(max_output_size, box_num)
    sorted_indices = sorted(range(len(scores)), key=lambda k: -scores[k])
    selected = []
    for i in range(box_num):
        if len(selected) >= output_size:
            break
        should_select = True
        for j in range(len(selected)-1,-1,-1):
            if polygon_iou(boxes[sorted_indices[i]], boxes[selected[j]])[0] > iou_thresh:
                should_select = False
                break
        if should_select:
            selected.append(sorted_indices[i])
        
    return np.array(selected, dtype=np.int32)

if __name__ == '__main__':
    import timeit
    rect1 = [(1,1), (0,1), (0,0), (1,0)]
    rect2 = [(1.25,2.25), (0.25,1.25), (1.25,0.25), (2.25,1.25)]
    print(timeit.timeit('polygon_iou(rect1, rect2)', number=10000, globals=globals())) 

