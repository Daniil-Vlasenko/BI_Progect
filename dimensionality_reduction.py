"""
This module contains the functions needed to convert the original NIfTI data from 4d to 2d and from 2d to 4d and
reduce the data.
"""
import nilearn.image as image
import nibabel as nib
import numpy as np


def resample(input_files, n, output_files):
    """
    Resample original voxel size of input files to n and save new NIfTI files.

    :param input_files: paths of original NIfTI files
    :type input_files: list
    :param n: new voxel's size
    :type n: int
    :param output_files: paths of new NIfTI files
    :type output_files: list
    """
    for i in range(len(input_files)):
        print(i)
        img = image.load_img(input_files[i])
        affine = np.diag((n, n, n))
        new_img = image.resample_img(img, target_affine=affine)
        nib.save(new_img, output_files[i])


def _4D_to_2D(data):
    """
    Converting 4D array to array of arrays that contain voxel values at different points in time.

    :param data: 4D array
    :type data: numpy.ndarray
    :return: array of arrays that contain voxel values at different points in time.
    :rtype: numpy.ndarray
    """

    return data.transpose(3, 0, 1, 2).reshape(data.shape[3], -1).transpose(1, 0)


def _2D_to_4D(data, shape):
    """
    Converting array of arrays that contain voxel values at different points in time to 4D array.

    :param data: array of arrays that contain voxel values at different points in time
    :type data: numpy.ndarray
    :param shape: shape of the new 4D array
    :type shape: list
    :type data: numpy.ndarray
    :return: array of arrays that contain voxel values at different points in time.
    :rtype: numpy.ndarray
    """

    return data.reshape(shape)
