"""
.. module:: nms
   :synopsis: NMS functions for [rects], [rotated_rects] and [polygons]

.. moduleauthor:: tom hoag <tomhoag@gmail.com>

These are the NMS functions you are looking for.

"""

import cv2

import nms.fast as fast
import nms.felzenszwalb as felzenszwalb
import nms.malisiewicz as malisiewicz


nms_algorithms = [felzenszwalb.nms, malisiewicz.nms, fast.nms]

default_algorithm = malisiewicz.nms


def rboxes(rrects, scores, nms_algorithm=default_algorithm, **kwargs):
    """
    Non Maxima Suppression for rotated rectangles

    :param rrects: a list of polygons, each described by ((cx, cy), (w,h), deg)
    :type rrects: list
    :param scores: a list of the scores associated with the rects
    :type scores: list
    :param nms_algorithm: the NMS comparison function to use, kwargs will be passed to this function. Defaults to :func:`nms.malisiewicz.NMS`
    :type nms_algorithm: function
    :returns: an array of indicies of the best rrects
    """

    # convert the rrects to polys
    polys = []
    for rrect in rrects:
        r = cv2.boxPoints(rrect)
        print(r)
        polys.append(r)

    return polygons(polys, scores, nms_algorithm, **kwargs)


def polygons(polys, scores, nms_algorithm=default_algorithm, **kwargs):
    """
    Non Maxima Suppression for polygons

    :param polys: a list of polygons, each described by their xy verticies
    :type polys: list
    :param scores: a list of the scores associated with the polygons
    :type scores: list
    :param nms_algorithm: the NMS comparison function to use, kwargs will be passed to this function. Defaults to :func:`nms.malisiewicz.nms`
    :type nms_algorithm: function
    :returns: an array of indicies of the best polys
    """

    assert nms_algorithm in nms_algorithms

    if nms_algorithm == fast.nms:
        kwargs['compare_function'] = fast.polygon_iou

    if nms_algorithm == felzenszwalb.nms:
        kwargs['area_function'] = felzenszwalb.poly_areas
        kwargs['compare_function'] = felzenszwalb.poly_compare

    if nms_algorithm == malisiewicz.nms:
        kwargs['area_function'] = malisiewicz.poly_areas
        kwargs['compare_function'] = malisiewicz.poly_compare

    return nms_algorithm(polys, scores, **kwargs)


def boxes(rects, scores, nms_algorithm=default_algorithm, **kwargs):
    """
    Non Maxima Suppression for rectangles.

    This function is provided for completeness as it replicates the functionality of cv2.dnn.NMSBoxes.  This *may* be
    slightly faster as NMSBoxes uses the FAST comparison algorithm and by default this used Malisiewicz et al.

    :param rects: a list of rectangles, each described by (x, y, w, h) (same as cv2.NMSBoxes)
    :type rects: list
    :param scores: a list of the scores associated with rects
    :type scores: list
    :param nms_algorithm: the NMS comparison function to use, kwargs will be passed to this function. Defaults to :func:`nms.malisiewicz.nms`
    :type nms_algorithm: function
    :returns: a list of indicies of the best rects
    """

    assert nms_algorithm in nms_algorithms

    if nms_algorithm == fast.nms:
        kwargs['compare_function'] = fast.rectangle_iou

    if nms_algorithm == felzenszwalb.nms:
        kwargs['area_function'] = felzenszwalb.rect_areas
        kwargs['compare_function'] = felzenszwalb.rect_compare

    if nms_algorithm == malisiewicz.nms:
        kwargs['area_function'] = malisiewicz.rect_areas
        kwargs['compare_function'] = malisiewicz.rect_compare

    return nms_algorithm(rects, scores, **kwargs)