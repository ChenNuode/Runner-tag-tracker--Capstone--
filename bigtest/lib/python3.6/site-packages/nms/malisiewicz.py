"""
.. module:: malisiewicz
   :synopsis: Malisiewicz et al implementation of NMS

.. moduleauthor:: tom hoag <tomhoag@gmail.com>

Malisiewicz et al implementation of NMS

The functions in this module are not usually called directly.  Instead use :func:`nms.nms.boxes`,
:func:`nms.nms.rboxes`, or :func:`nms.nms.polygons`


"""
import numpy as np
import cv2
import nms.helpers as help


def rect_areas(rects):
    """Return an np.array of the areas of the rectangles

        :param rects: a list of rectangles, each specified as (x, y, w, h)
        :type rects: list
        :return: an numpy array of corresponding areas
        :rtype: :class:`numpy.ndarray`
        """
    # rect = x,y,w,h
    w = rects[:,2]
    h = rects[:,3]
    return w * h

def rect_compare(box, boxes, area):
    """Calculate the intersection of box to boxes divided by area

    :param box: a rectangle specified by (x, y, w, h)
    :type box: tuple
    :param boxes: a list of rectangles, each specified by (x, y, w, h)
    :type boxes: list
    :param area: a list of areas of the corresponding boxes
    :type area: list
    :return: a numpy array of the ratio of overlap of box to each of boxes to the corresponding area.  e.g. overlap(box, boxes[n])/area[n]
    :rtype: :class:`numpy.ndarray`
    """
    # box and boxes are XYWH
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    x2 = x1 + w
    y2 = y1 + h

    # find the largest (x, y) coordinates for the start of
    # the bounding box and the smallest (x, y) coordinates
    # for the end of the bounding box
    xx1 = np.maximum(box[0], x1)
    yy1 = np.maximum(box[1], y1)
    xx2 = np.minimum(box[0] + box[2], x2)
    yy2 = np.minimum(box[1] + box[3], y2)

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    return (w * h)/area


def poly_areas(polys):
    """Calculate the area of the list of polygons

    :param polys: a list of polygons, each specified by a list of its verticies
    :type polys: list
    :return: numpy array of areas of the polygons
    :rtype: :class:`numpy.ndarray`
    """
    areas = []
    for poly in polys:
        areas.append(cv2.contourArea(np.array(poly, np.int32)))
    return np.array(areas)


def poly_compare(poly1, polygons, area):
    """Calculate the intersection of poly1 to polygons divided by area

    :param poly1: a polygon specified by a list of its verticies
    :type poly1: list
    :param polygons: a list of polygons, each specified a list of its verticies
    :type polygons: list
    :param area: a list of areas of the corresponding polygons
    :type area: list
    :return: a numpy array of the ratio of overlap of poly1 to each of polygons to the corresponding area.  e.g. overlap(poly1, polygons[n])/area[n]
    :rtype: :class:`numpy.ndarray`
    """
    # return intersection of poly1 with polys[i]/area[i]
    overlap = []
    for i,poly2 in enumerate(polygons):
        intersection_area = help.polygon_intersection_area([poly1, poly2])
        overlap.append(intersection_area/area[i])

    return np.array(overlap)


def nms(boxes, scores, **kwargs):
    """NMS using Malisiewicz et al. method

    Adapted from  non_max_suppression_fast(boxes, overlapThresh) from
    `(Faster) Non-Maximum Suppression in Python <https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/>`_

    Note that when using this on rotated rectangles or polygons (as opposed to upright rectangles), some of the
    vectorized performance gains are lost as comparisons between rrects and polys cannot(?) be vectorized

    This function is not usually called directly.  Instead use :func:`nms.nms.boxes`, :func:`nms.nms.rboxes`,
    or :func:`nms.nms.polygons` and set parameter `nms_algorithm=nms.malisiewicz`

    :param boxes: a list of boxes to perform NMS on
    :type boxes: list
    :param scores: a list of scores corresponding to boxes
    :type scores: list
    :param kwargs: optional keyword parameters (see below)
    :type kwargs: dict
    :returns: a list of the indicies of the best boxes

    :kwargs:

    - top_k (int): if >0, keep at most top_k picked indices. default:0
    - score_threshold (float): the minimum score necessary to be a viable solution, default 0.3
    - nms_threshold (float): the minimum nms value to be a viable solution, default: 0.4
    - compare_function (function): function that accepts two boxes and returns their overlap ratio,
      this function must accept two boxes and return an overlap ratio between 0 and 1
    - area_function (function): function used to calculate the area of an element of boxes
    """

    if 'top_k' in kwargs:
        top_k = kwargs['top_k']
    else:
        top_k = 0
    assert 0 <= top_k

    if 'score_threshold' in kwargs:
        score_threshold = kwargs['score_threshold']
    else:
        score_threshold = 0.3
    assert 0 < score_threshold

    if 'nms_threshold' in kwargs:
        nms_threshold = kwargs['nms_threshold']
    else:
        nms_threshold = 0.4
    assert 0 < nms_threshold < 1

    if 'compare_function' in kwargs:
        compare_function = kwargs['compare_function']
    else:
        compare_function = None
    assert compare_function is not None

    if 'area_function' in kwargs:
        area_function = kwargs['area_function']
    else:
        area_function = None
    assert area_function is not None

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    if scores is not None:
        assert len(scores) == len(boxes)

    boxes = np.array(boxes)

    if compare_function == rect_compare:
        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    areas = area_function(boxes) #(x2 - x1 + 1) * (y2 - y1 + 1)

    # sort the boxes by score or the bottom-right y-coordinate of the bounding box
    if scores is not None:
        # sort the bounding boxes by the associated scores
        scores = help.get_max_score_index(scores, score_threshold, top_k, False)
        idxs = np.array(scores, np.int32)[:, 1]
        # idxs = np.argsort(scores)
    else:
        # sort the bounding boxes by the bottom-right y-coordinate of the bounding box
        y2 = boxes[:3]
        idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    #boxes = np.array(boxes)
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # compute the ratio of overlap
        overlap = compare_function(boxes[i], boxes[idxs[:last]], areas[idxs[:last]])
        #(w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > nms_threshold)[0])))

    # return the indicies of the picked bounding boxes that were picked
    return pick