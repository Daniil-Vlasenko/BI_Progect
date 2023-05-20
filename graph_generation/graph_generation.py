"""
This module contains the functions needed to reduce time dimension and calculate a^T,
calculate edge weights and generate graphs.
"""
import glob
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import igraph as ig
import nibabel as nib
from dimensionality_reduction import dimensionality_reduction as dr
from classifier_learning import classifier_learning as cl


def scalarization_1(data, mode, q_1=0.1, q_2=0.9):
    """
    Convert time series for each voxel to scalar value.

    :param data: array of arrays that contain voxel's values at different points in time.
    :type data: numpy.ndarray
    :param mode: T function; 0 - mean, 1 - median, 2 -standard derivation, 3 - variance, 4 - difference between the maximum and minimum, 5 - difference between quantiles, 6 - maximum, 7 - minimum, 8 - quantile 1, 9 - quantile 2
    :type mode: int
    :param q_1: level of quantile 1
    :type q_1: float
    :param q_2: level of quantile 2
    :type q_2: float
    :return: array of scalars.
    :rtype: numpy.ndarray
    """
    if mode == 0:
        return np.mean(data, axis=1)
    elif mode == 1:
        return np.median(data, axis=1)
    elif mode == 2:
        return np.std(data, axis=1)
    elif mode == 3:
        return np.var(data, axis=1)
    elif mode == 4:
        return np.max(data, axis=1) - np.min(data, axis=1)
    elif mode == 5:
        return np.quantile(data, q_2, axis=1) - np.quantile(data, q_1, axis=1)
    elif mode == 6:
        return np.max(data, axis=1)
    elif mode == 7:
        return np.min(data, axis=1)
    elif mode == 8:
        return np.quantile(data, q_1, axis=1)
    elif mode == 9:
        return np.quantile(data, q_2, axis=1)
    else:
        raise Exception("Invalid mode.")


def scalarization_2(input_file, mode, q_1=0.1, q_2=0.9):
    """
    Convert time series for each voxel to scalar value.

    :param input_file: path to NIFTI data.
    :type input_file: str
    :param mode: T function; 0 - mean, 1 - median, 2 -standard derivation, 3 - variance, 4 - difference between the maximum and minimum, 5 - difference between quantiles, 6 - maximum, 7 - minimum, 8 - quantile 1, 9 - quantile 2
    :type mode: int
    :param q_1: level of quantile 1
    :type q_1: float
    :param q_2: level of quantile 2
    :type q_2: float
    :return: array of scalars
    :rtype: numpy.ndarray
    """
    img = nib.load(input_file)
    data = img.get_fdata()
    data = dr._4D_to_2D(data)
    return scalarization_1(data, mode, q_1, q_2)


def scalarization_3(input_files, output_file, mode, q_1=0.1, q_2=0.9):
    """
    Convert time series for each voxel to scalar value and save results.

    :param input_files: paths to NIFTI data
    :type input_files: list
    :param output_file: path of output file
    :type output_file: str
    :param mode: T function; 0 - mean, 1 - median, 2 -standard derivation, 3 - variance, 4 - difference between the maximum and minimum, 5 - difference between quantiles, 6 - maximum, 7 - minimum, 8 - quantile 1, 9 - quantile 2
    :type mode: int
    :param q_1: level of quantile 1
    :type q_1: float
    :param q_2: level of quantile 2
    :type q_2: float
    :return: array of scalars
    :rtype: numpy.ndarray
    """
    data = scalarization_2(input_files[0], mode)
    size = len(input_files)
    for i in range(1, size):
        tmp = scalarization_2(input_files[i], mode, q_1, q_2)
        data = np.vstack((data, tmp))
    data = np.transpose(data)
    np.savetxt(output_file, data)
    return data


def edges_calculation(classifiers_folder, scalars_file, shape, edges_file):
    """
    Calculate edge weight for every two adjacent voxels of one scalar type for both perception and imagery regimes
    for every run of fMRI.

    :param classifiers_folder: path to the folder where classifiers are
    :type classifiers_folder: str
    :param scalars_file: path to the file of scalars
    :type scalars_file: str    
    :param shape: shape of the 3D array a^T
    :type shape: list
    :param edges_file: path to the file of new edge weights
    :type edges_file: str    .
    """
    scalars_np = np.loadtxt(scalars_file)

    if len(scalars_np.shape) == 1:
        number_of_runs = 1
    else:
        number_of_runs = len(scalars_np[0])

    number_of_edges = int((26 * (shape[0] - 2) * (shape[1] - 2) * (shape[2] - 2) + 8 * 19 +
                           4 * ((shape[0] - 2) + (shape[1] - 2) + (shape[2] - 2) - 6) * 15 +
                           2 * ((shape[0] - 2) * (shape[1] - 2) + (shape[0] - 2) * (shape[2] - 2) + (shape[1] - 2) * 
                           (shape[2] - 2) - 4 * (shape[0] - 2) - 4 * (shape[1] - 2) - 4 * (shape[2] - 2) + 12) * 9) / 2)

    edges_np = np.zeros((number_of_edges, number_of_runs + 2))    

    count_of_rows = 0
    neighbors = cl.get_set_of_neighbors(shape)
    for pair in neighbors:
        voxel_1_id = pair[0]
        voxel_2_id = pair[1]

        print("count:", count_of_rows + 1, "voxel_1_id:", voxel_1_id, "voxel_2_id:", voxel_2_id)
        if "GPC" in classifiers_folder:
            classifier_file = classifiers_folder + "/GPC_voxel_" + str(voxel_1_id) + "_and_voxel_" + \
                              str(voxel_2_id) + ".sav"
            classifier = joblib.load(classifier_file)
        else:
            classifier_file = classifiers_folder + "/SVC_voxel_" + str(voxel_1_id) + "_and_voxel_" + \
                              str(voxel_2_id) + ".sav"
            classifier = pickle.load(open(classifier_file, 'rb'))

        edges_np[count_of_rows][0] = voxel_1_id
        edges_np[count_of_rows][1] = voxel_2_id

        if number_of_runs == 1:
            tmp = pd.DataFrame(
                {'voxel1': [scalars_np[voxel_1_id]], 'voxel2': [scalars_np[voxel_2_id]]})
        else:
            tmp = pd.DataFrame(
                {'voxel1': scalars_np[voxel_1_id], 'voxel2': scalars_np[voxel_2_id]})

        tmp = classifier.predict_proba(tmp)
        edges_np[count_of_rows][2:] = tmp[:, 1] - tmp[:, 0]    

        count_of_rows += 1

    column_per_names = ["sours", "target"] + [str(i) for i in range(number_of_runs)]    
    edges_df_per = pd.DataFrame(data=edges_np, columns=column_per_names)    
    edges_df_per = edges_df_per.astype({"sours": "int", "target": "int"})    
    edges_df_per.to_csv(edges_file, index=False)


def graphs_generation(scalars_file, shape, edges_file, graph_folder):
    """
    Generation of graphs.

    :param scalars_file: path to the file of scalars
    :type scalars_file: str
    :param shape: shape of the 3D array a^T
    :type shape: list
    :param edges_file: path to the file of edge weights
    :type edges_file: str
    :param graph_folder: path to the folder where graphs will be saved
    :type graph_folder: str    
    """
    scalars_np = np.loadtxt(scalars_file)
    edges_df = pd.read_csv(edges_file)    

    if len(scalars_np.shape) == 1:
        number_of_runs = 1
    else:
        number_of_runs = len(scalars_np[0])

    number_of_voxels = len(scalars_np)
    id_vertices = [i for i in range(number_of_voxels)]
    id_x_vertices = [np.unravel_index(i, shape[:3])[0] for i in range(number_of_voxels)]
    id_y_vertices = [np.unravel_index(i, shape[:3])[1] for i in range(number_of_voxels)]
    id_z_vertices = [np.unravel_index(i, shape[:3])[2] for i in range(number_of_voxels)]

    for i in range(number_of_runs):
        edges_df_new = edges_df[["sours", "target", str(i)]]
        edges_df_new = edges_df_new.rename(
            columns={"sours": "sours", "target": "target", str(i): "value"})
        if len(scalars_np.shape) == 1:
            vertices_df = pd.DataFrame({'flat_id_voxel': id_vertices, "x_id": id_x_vertices,
                                                   "y_id": id_y_vertices, "z_id": id_z_vertices,
                                                   "value": scalars_np})
        else:
            vertices_df = pd.DataFrame({'flat_id_voxel': id_vertices, "x_id": id_x_vertices,
                                                   "y_id": id_y_vertices, "z_id": id_z_vertices,
                                                   "value": scalars_np[:, i]})

        g = ig.Graph.DataFrame(edges_df_new, directed=False, vertices=vertices_df)
        file_name = graph_folder + "/run_" + str(i) + ".gml"
        g.write(file_name, format="gml")
        print(i)


def graph_delete(graph_folder):
    """
    Delete all graphs in folder.

    :param graph_folder: path to the folder of graphs
    :type graph_folder: str
    """
    files = glob.glob(graph_folder + '/*')
    for f in files:
        os.remove(f)
