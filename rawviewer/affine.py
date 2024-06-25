''' affine.py '''
__version__ = "0.0.1"
# ======================================================================
# Import
# ======================================================================
import cv2
import numpy as np


# ======================================================================
# Global
# ======================================================================
AFFINE_MATRIX_SIZE = 3  ## size of affine matrix = 3 x 3


# ======================================================================
# Functions
# ======================================================================
def identityMatrix(n :int):
    '''n x n identityMatrix

    Parameters
    ----------
    n : int
        Size of identityMatrix

    Returns
    -------
    identityMatrix : numpy.ndarray([n, n], dtype=float32)
        generated identityMatrix
    '''
    return np.eye(n, dtype = np.float32)


def scaleMatrix(scale_x: float, scale_y: float):
    '''affine matrix for scale

    Parameters
    ----------
    scale_x : numpy.float32
        scale of x-axis
    scale_y : numpy.float32
        scale of y-axis

    Returns
    -------
    affine_scale_matrix : numpy.ndarray([3, 3], dtype=float32)
        generated matrix
    '''
    mat = identityMatrix(AFFINE_MATRIX_SIZE)
    mat[0, 0] = scale_x
    mat[1, 1] = scale_y

    return mat


def translateMatrix(translate_x: float, translate_y: float):
    '''affine matrix for translate

    Parameters
    ----------
    translate_x : numpy.float32
        translate of x-axis
    translate_y : numpy.float32
        translate of y-axis

    Returns
    -------
    affine_translate_matrix : numpy.ndarray([3, 3], dtype=float32)
        generated matrix
    '''
    mat = identityMatrix(AFFINE_MATRIX_SIZE)
    mat[0, 2] = translate_x
    mat[1, 2] = translate_y

    return mat


def rotateMatrix(deg: float):
    '''affine matrix for rotate

    Parameters
    ----------
    deg : numpy.float32
        rotate degree

    Returns
    -------
    affine_rotate_matrix : numpy.ndarray([n, n], dtype=float32)
        generated matrix
    '''
    mat = identityMatrix(AFFINE_MATRIX_SIZE)
    rad = np.deg2rad(deg)
    sin = np.sin(rad)
    cos = np.cos(rad)

    mat[0, 0] = cos
    mat[0, 1] = -sin
    mat[1, 0] = sin
    mat[1, 1] = cos

    return mat


def scaleAtMatrix(scale_x, scale_y, center_x, center_y):
    '''affine matrix for scale (based on center position)

    Parameters
    ----------
    scale_x : numpy.float32
        scale of x-axis
    scale_y : numpy.float32
        scale of y-axis
    center_x : numpy.float32
        center position of x-axis
    center_y : numpy.float32
        center position of y-axis

    Returns
    -------
    affine_scale_matrix : numpy.ndarray([3, 3], dtype=float32)
        generated matrix
    '''
    mat_center       = translateMatrix(-center_x, -center_y)                     ## 基点の座標を原点に移動
    mat_center_scale = scaleMatrix(scale_x, scale_y).dot(mat_center)             ## 原点周りに拡大縮小
    mat              = translateMatrix(center_x, center_y).dot(mat_center_scale) ## 元の位置に戻す

    return mat


def rotateAtMatrix(deg, center_x, center_y):
    '''affine matrix for rotate (based on center position)

    Parameters
    ----------
    deg : numpy.float32
        rotate degree
    center_x : numpy.float32
        center position of x-axis
    center_y : numpy.float32
        center position of y-axis

    Returns
    -------
    affine_rotate_matrix : numpy.ndarray([3, 3], dtype=float32)
        generated matrix
    '''
    mat_center        = translateMatrix(-center_x, -center_y)                      ## 基点の座標を原点に移動
    mat_center_rotate = rotateMatrix(deg).dot(mat_center)                          ## 原点周りに拡大縮小
    mat               = translateMatrix(center_x, center_y).dot(mat_center_rotate) ## 元の位置に戻す

    return mat


def affinePoint(affine_mat, x, y):
    '''affine transformed coordinate

    Parameters
    ----------
    affine_mat : numpy.ndarray([3, 3], dtype = numpy.float32)
        affine matrix
    x : numpy.float32
        center position of x-axis
    y : numpy.float32
        center position of y-axis

    Returns
    -------
    affine_coordinate_x : float
        affine transformed coordinate (x-axis)
    affine_coordinate_y : float
        affine transformed coordinate (y-axis)
    '''
    src_point          = np.array([x, y, 1])
    x_affine, y_affine = affine_mat.dot(src_point)[:2]

    return x_affine, y_affine


def inverse(mat):
    '''inverse matrix

    Parameters
    ----------
    mat : numpy.ndarray([3, 3], dtype = numpy.float32)
        matrix

    Returns
    -------
    affine_rotate_matrix : numpy.ndarray([3, 3], dtype=float32)
        generated matrix
    '''
    return np.linalg.pinv(mat)

