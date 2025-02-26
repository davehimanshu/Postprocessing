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

#-------------------------------------------------------------------------------#
# Obtain data from the means.h5 file
#-------------------------------------------------------------------------------#
def extract_data_from_group(file_path, group_name):
    """
    Extracts data from a specified group in an HDF5 file.

    Parameters:
    file_path (str): Path to the HDF5 file.
    group_name (str): Name of the group to extract data from.

    Returns:
    dict: A dictionary containing datasets from the specified group.
    """
    data_dict = {}
    try:
        with h5py.File(file_path, 'r') as h5_file:
            if group_name in h5_file:
                group = h5_file[group_name]
                for dataset_name in group:
                    data_dict[dataset_name] = group[dataset_name][:]
            else:
                print(f"Group '{group_name}' not found in the file.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return data_dict
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Calculate the root mean square fluctuations
#-------------------------------------------------------------------------------#
def calculate_rms_fluctuations(data_mean, data_rms):
    # Ensure the keys are the same in both dictionaries
    common_keys = set(data_mean.keys()).intersection(data_rms.keys())
    data_result = {}
    for key in common_keys:
        data_result[key] = np.sqrt(data_rms[key]**2 - data_mean[key]**2)
    return data_result
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Calculate interface height
#-------------------------------------------------------------------------------#
def calc_interface_height(phibar, xmr):
    # Obtain list of keys
    keys = list(phibar.keys())
    result = np.zeros(len(keys))
    for key in keys:
        phibar[key] = phibar[key] - 0.5
        sign_changes = np.where(np.diff(np.sign(phibar[key])))[0]
        if sign_changes.size > 0:
            result[keys.index(key)] = xmr[sign_changes[0]]
        else:
            result[keys.index(key)] = 0.0  # or some other default value if no sign change is found
    return result
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Calculate effective Rayleigh number
#-------------------------------------------------------------------------------#
def calc_effective_rayleigh(hbar, ra_final):
    # Calculate the Rayleigh number
    ra = np.zeros(len(hbar))
    for i in range(len(hbar)):
        ra[i] = ra_final * hbar[i]**3
    return ra
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Calculate effective Ekman number
#-------------------------------------------------------------------------------#
def calc_effective_ekman_number(hbar, Rossby_final, Ra_final, Pr):
    # first calculate the Ekman number
    Ek = Rossby_final/np.sqrt(Ra_final/Pr)
    # Calculate the effective Ekman number
    ek_eff = np.zeros(len(hbar))
    for i in range(len(hbar)):
        ek_eff[i] = Ek / (hbar[i]**2)
    return ek_eff
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Calulate the effective Rossby number
#-------------------------------------------------------------------------------#
def calc_effective_rossby_number(ek_eff, ra_eff, Pr):
    # Calculate the effective Rossby number
    ross_eff = np.zeros(len(ek_eff))
    for i in range(len(ek_eff)):
        ross_eff[i] = ek_eff[i] * np.sqrt(ra_eff[i]/Pr)
    return ross_eff
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Calculate the Nusselt number
#-------------------------------------------------------------------------------#
#### INSERT SUBROUTINE HERE ####
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Calculate fft for 3d fields
#-------------------------------------------------------------------------------#
#### INSERT SUBROUTINE HERE ####
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Save data as text file with n columns
#-------------------------------------------------------------------------------#
def save_data_to_text(folder_path, file_name, num_columns, headers, *arrays):
    """
    Saves data as a text file with the specified number of columns and headers.

    Parameters:
    folder_path (str): Path to the folder where the text file will be saved.
    file_name (str): Name of the text file.
    num_columns (int): Number of columns in the text file.
    headers (list): List of headers for the columns.
    *arrays (list of np.ndarray): Arrays to be filled inside these columns.

    Returns:
    None
    """
    if len(headers) != num_columns:
        raise ValueError("Number of headers must match the number of columns.")
    if len(arrays) != num_columns:
        raise ValueError("Number of arrays must match the number of columns.")
    if not all(len(arr) == len(arrays[0]) for arr in arrays):
        raise ValueError("All arrays must have the same length.")

    data = np.column_stack(arrays)
    file_path = os.path.join(folder_path, file_name)
    
    with open(file_path, 'w') as f:
        # Write headers
        f.write('\t'.join(headers) + '\n')
        # Write data
        np.savetxt(f, data, delimiter='\t', fmt='%s')