import os
import h5py
import numpy as np


#-------------------------------------------------------------------------------#
# This function reads the coordinate information from the 
# cord_info.h5 file which stores the coordinate information as a dictionary.
#-------------------------------------------------------------------------------#
def read_cord_info(directory_path, file_extension='.h5', specific_file_name=None):
    if specific_file_name:
        h5_files = [specific_file_name] if specific_file_name.endswith(file_extension) else []
    else:
        h5_files = [f for f in os.listdir(directory_path) if f.endswith(file_extension)]
    
    data_dict = {}
    for h5_file in h5_files:
        file_path = os.path.join(directory_path, h5_file)
        with h5py.File(file_path, 'r') as h5_file:
            for key in h5_file.keys():
                data = h5_file[key][:]
                data_dict[key] = data
    return data_dict
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# This function reads data from the h5 file within the outputdir/fields folder
#-------------------------------------------------------------------------------#
def read_h5_file_fields(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        # Get the first (and only) dataset name
        dataset_name = list(h5_file.keys())[0]
        # Get the data
        data = h5_file[dataset_name][:]
        return data
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# This function creates an array of file numbers and times that are stored in the 
# outputdir/fields folder specifically useful for AFiD-MuRPhFi simulation files
#-------------------------------------------------------------------------------#
def generate_time_and_files_fields(SAVE_3D, t_time):
    t_int = np.arange(0, t_time + SAVE_3D, SAVE_3D)
    steps = int(t_time / SAVE_3D)
    file_analyze = np.linspace(0, steps, steps + 1, dtype=int)
    file_analyze = [f"{num:05d}" for num in file_analyze]
    return t_int, file_analyze
#-------------------------------------------------------------------------------#
