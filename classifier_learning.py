"""
This module contains the functions needed to teach classifiers Cl_{ij} for calculation of edges weights.
"""
import joblib
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import pickle
from sklearn.svm import SVC
import numpy as np
import pandas as pd


def get_set_of_neighbors(shape):
    """
    Calculation of set of pairs of adjacent voxels.

    :param shape: shape of the 3D array a^T
    :type shape: list
    :return: set of pairs of adjacent voxels
    :rtype: set
    """
    set_ = set()
    for x in range(1, shape[0] - 1):
        for y in range(1, shape[1] - 1):
            for z in range(1, shape[2] - 1):
                voxel_id = np.ravel_multi_index((x, y, z), shape[:3])
                neighbors = [tuple(sorted([np.ravel_multi_index((x - 1, y - 1, z - 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x - 1, y, z - 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x - 1, y + 1, z - 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x, y - 1, z - 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x, y, z - 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x, y + 1, z - 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x + 1, y - 1, z - 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x + 1, y, z - 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x + 1, y + 1, z - 1), shape[:3]), voxel_id])),

                             tuple(sorted([np.ravel_multi_index((x - 1, y - 1, z + 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x - 1, y, z + 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x - 1, y + 1, z + 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x, y - 1, z + 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x, y, z + 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x, y + 1, z + 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x + 1, y - 1, z + 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x + 1, y, z + 1), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x + 1, y + 1, z + 1), shape[:3]), voxel_id])),

                             tuple(sorted([np.ravel_multi_index((x - 1, y - 1, z), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x - 1, y, z), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x - 1, y + 1, z), shape[:3]), voxel_id])),

                             tuple(sorted([np.ravel_multi_index((x + 1, y - 1, z), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x + 1, y, z), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x + 1, y + 1, z), shape[:3]), voxel_id])),

                             tuple(sorted([np.ravel_multi_index((x, y - 1, z), shape[:3]), voxel_id])),
                             tuple(sorted([np.ravel_multi_index((x, y + 1, z), shape[:3]), voxel_id]))]
                set_.update(neighbors)
    return set_


def SVM_learning(mode_1_s_file, mode_2_s_file, shape, classifiers_folder):
    """
    Teaching of SVM classifiers for calculation of edges weights.

    :param mode_1_s_file: path to file of scalars a^T of mode 1.
    :type mode_1_s_file: str
    :param mode_2_s_file: path to file of scalars a^T of mode 2.
    :type mode_2_s_file: str
    :param shape: shape of the 3D array a^T
    :type shape: list
    :param classifiers_folder: path to folder of classifiers
    :type classifiers_folder: str
    """
    training_mode_1_file = open(mode_1_s_file, "r")
    training_mode_2_file = open(mode_2_s_file, "r")

    training_perception_lines = training_mode_1_file.readlines()
    training_imagery_lines = training_mode_2_file.readlines()

    count = 0
    neighbors = get_set_of_neighbors(shape)
    for pair in neighbors:
        voxel_1_id = pair[0]
        voxel_2_id = pair[1]

        count += 1
        print("count:", count, "voxel_1_id:", voxel_1_id, "voxel_2_id:", voxel_2_id)

        training_perception_voxel1 = np.fromstring(training_perception_lines[voxel_1_id], dtype=float, sep=' ')
        training_imagery_voxel1 = np.fromstring(training_imagery_lines[voxel_1_id], dtype=float, sep=' ')
        training_perception_voxel2 = np.fromstring(training_perception_lines[voxel_2_id], dtype=float, sep=' ')
        training_imagery_voxel2 = np.fromstring(training_imagery_lines[voxel_2_id], dtype=float, sep=' ')

        training_voxel1 = np.concatenate((training_perception_voxel1, training_imagery_voxel1))
        training_voxel2 = np.concatenate((training_perception_voxel2, training_imagery_voxel2))

        training_X = pd.DataFrame({'voxel1': training_voxel1, 'voxel2': training_voxel2})
        training_y = [1 for i in range(len(training_perception_voxel1))] + \
                     [2 for j in range(len(training_imagery_voxel1))]

        svc = SVC(kernel="rbf", C=1, probability=True)
        svc.fit(training_X, training_y)
        filename = classifiers_folder + "/SVC_voxel_" + str(voxel_1_id) + "_and_voxel_" + str(voxel_2_id) + ".sav"
        pickle.dump(svc, open(filename, 'wb'))


def GPC_learning(mode_1_s_file, mode_2_s_file, shape, classifiers_folder):
    """
    Teaching of GPC classifiers for calculation of edges weights.

    :param mode_1_s_file: path to file of scalars a^T of mode 1.
    :type mode_1_s_file: str
    :param mode_2_s_file: path to file of scalars a^T of mode 2.
    :type mode_2_s_file: str
    :param shape: shape of the 3D array a^T
    :type shape: list
    :param classifiers_folder: path to folder of classifiers
    :type classifiers_folder: str
    """
    training_mode_1_file = open(mode_1_s_file, "r")
    training_mode_2_file = open(mode_2_s_file, "r")

    training_perception_lines = training_mode_1_file.readlines()
    training_imagery_lines = training_mode_2_file.readlines()

    count = 0
    neighbors = get_set_of_neighbors(shape)
    for pair in neighbors:
        voxel_1_id = pair[0]
        voxel_2_id = pair[1]

        count += 1
        print("count:", count, "voxel_1_id:", voxel_1_id, "voxel_2_id:", voxel_2_id)

        training_perception_voxel1 = np.fromstring(training_perception_lines[voxel_1_id], dtype=float, sep=' ')
        training_imagery_voxel1 = np.fromstring(training_imagery_lines[voxel_1_id], dtype=float, sep=' ')
        training_perception_voxel2 = np.fromstring(training_perception_lines[voxel_2_id], dtype=float, sep=' ')
        training_imagery_voxel2 = np.fromstring(training_imagery_lines[voxel_2_id], dtype=float, sep=' ')

        training_voxel1 = np.concatenate((training_perception_voxel1, training_imagery_voxel1))
        training_voxel2 = np.concatenate((training_perception_voxel2, training_imagery_voxel2))

        training_X = pd.DataFrame({'voxel1': training_voxel1, 'voxel2': training_voxel2})
        training_y = [1 for i in range(len(training_perception_voxel1))] + \
                     [2 for j in range(len(training_imagery_voxel1))]

        gpc = GaussianProcessClassifier(kernel=RBF(1))
        gpc.fit(training_X, training_y)
        filename = classifiers_folder + "/GPC_voxel_" + str(voxel_1_id) + "_and_voxel_" + str(voxel_2_id) + ".sav"
        joblib.dump(gpc, filename)
