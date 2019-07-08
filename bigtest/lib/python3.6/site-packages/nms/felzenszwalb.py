"""
.. module:: felzenszwalb
   :synopsis: Felzenszwalb et al implementation of NMS

.. moduleauthor:: tom hoag <tomhoag@gmail.com>

Felzenszwalb et al implementation of NMS

The functions in this module are not usually called directly.  Instead use :func:`nms.nms.boxes`,
:func:`nms.nms.rboxess`, or :func:`nms.nms.polygons`


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
    rects = np.array(rects)
    w = rects[:,2]
    h = rects[:,3]
    return w * h


def rect_compare(rect1, rect2, area):
    """Calculate the ratio of overlap between two rectangles and the given area

    :param rect1: rectangle specified as (x, y, w, h)
    :type rect1: tuple
    :param rect2: rectangle specificed as (x, y, w, h)
    :type rect2: tuple
    :param area: the area to compare to
    :type area: float
    :return: the ratio of the overlap of rect1 and rect2 to the area, e.g overlap(rect1, rect2)/area
    :rtype: float
    """
    # rect = x,y, w, h
    xx1 = max(rect1[0], rect2[0])
    yy1 = max(rect1[1], rect2[1])
    xx2 = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
    yy2 = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
    w = max(0, xx2 - xx1 + 1)
    h = max(0, yy2 - yy1 + 1)
    return float(w * h) / area


def poly_areas(polys):
    """Calculate the area of each polygon in polys

    :param polys: a list of polygons, each specified by its verticies
    :type polys: list
    :return: a list of areas corresponding the list of polygons
    :rtype: list
    """
    areas = []
    for poly in polys:
        areas.append(cv2.contourArea(np.array(poly, np.int32)))
    return areas


def poly_compare(poly1, poly2, area):
    """Calculate the ratio of overlap between two polygons and the given area

    :param poly1: polygon specified by its verticies
    :type poly1: list
    :param poly2: polygon specified by its verticies
    :type poly2: list
    :param area: the area to compare the overlap of poly1 and poly2
    :type area: float
    :return: the ratio of overlap of poly1 and poly2 to the area  e.g. overlap(poly1, poly2)/area
    :rtype: float
    """
    assert area > 0
    intersection_area = help.polygon_intersection_area([poly1, poly2])
    return intersection_area/area


def nms(boxes, scores, **kwargs):
    """NMS using Felzenszwalb et al. method

    Adapted from  non_max_suppression_slow(boxes, overlapThresh) from
    `Non-Maximum Suppression for Object Detection in Python <https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/>`_

    This function is not usually called directly.  Instead use :func:`nms.nms.boxes`, :func:`nms.nms.rboxes`,
     or :func:`nms.nms.polygons` and set `nms_algorithm=nms.felzenszwalb`

    :param boxes: a list of boxes to perform NMS on
    :type boxes: list
    :param scores: a list of scores corresponding to boxes
    :type scores: list
    :param kwargs: optional keyword parameters (see below)
    :type kwargs: dict (see below)
    :return: a list of the indicies of the best boxes
    :rtype: list

    :kwargs:

    - top_k (int): if >0, keep at most top_k picked indices. default:0, int
    - score_threshold (float): the minimum score necessary to be a viable solution, default 0.3, float
    - nms_threshold (float): the minimum nms value to be a viable solution, default: 0.4, float
    - compare_function (function): function that accepts two boxes and returns their overlap ratio, this function must
      accept two boxes and return an overlap ratio between 0 and 1
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

    # initialize the list of picked indexes
    pick = []

    # compute the area of the bounding boxes
    area = area_function(boxes)

    # sort the boxes by score or the bottom-right y-coordinate of the bounding box
    if scores is not None:
        # sort the bounding boxes by the associated scores
        scores = help.get_max_score_index(scores, score_threshold, top_k, False)
        idxs = np.array(scores, np.int32)[:,1]
        #idxs = np.argsort(scores)
    else:
        # sort the bounding boxes by the bottom-right y-coordinate of the bounding box
        y2 = boxes[:3]
        idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # compute the ratio of overlap between the two boxes and the area of the second box
            overlap = compare_function(boxes[i], boxes[j], area[j])

            # if there is sufficient overlap, suppress the current bounding box
            if overlap > nms_threshold:
                suppress.append(pos)

        # delete all indexes from the index list that are in the suppression list
        idxs = np.delete(idxs, suppress)

    # return only the indicies of the bounding boxes that were picked
    return pick
