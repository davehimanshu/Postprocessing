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
# Allows specifying which dataset to read from the file
#-------------------------------------------------------------------------------#
def read_h5_file_fields(file_path, dataset_name=None):
    with h5py.File(file_path, 'r') as h5_file:
        if dataset_name is None:
            # Get the first (and only) dataset name
            dataset_name = list(h5_file.keys())[0]
        if dataset_name not in h5_file:
            raise KeyError(f"Dataset '{dataset_name}' not found in file '{file_path}'")
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
# Obtain data from the means.h5 file (helpful since the file has multiple groups)
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
# Calculate the root mean square fluctuations from the means file
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
# Calculate the average interface height from the means file
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
# Calculate effective Rayleigh number from the means file
#-------------------------------------------------------------------------------#
def calc_effective_rayleigh(hbar, ra_final):
    # Calculate the Rayleigh number
    ra = np.zeros(len(hbar))
    for i in range(len(hbar)):
        ra[i] = ra_final * hbar[i]**3
    return ra
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Calculate effective Ekman number from the means file
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
# Calulate the effective Rossby number from the means file
#-------------------------------------------------------------------------------#
def calc_effective_rossby_number(ek_eff, ra_eff, Pr):
    # Calculate the effective Rossby number
    ross_eff = np.zeros(len(ek_eff))
    for i in range(len(ek_eff)):
        ross_eff[i] = ek_eff[i] * np.sqrt(ra_eff[i]/Pr)
    return ross_eff
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Calculate the interface position (2D) from the fields files
#-------------------------------------------------------------------------------#
def calc_interface_position_2d(phi, hori, vert):
    # Center around 0.0
    phi = phi - 0.5
    phi_2d = phi[0, :, :]
    vert_sign_change = np.zeros(len(hori))
    for j in range(len(hori)):
        for i in range(len(vert)):
            if phi_2d[j, i] > 0.0:
                vert_sign_change[j] = vert[i]
                break

    return hori, vert_sign_change
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Calculate the total liquid volume from the fields files
#-------------------------------------------------------------------------------#
def calc_total_liquid_volume(phi, cell_volume):
    # Center around 0.0
    phi_centered = phi - 0.5
    # Count number of cells where phi < 0 (liquid region)
    num_liquid_cells = np.sum(phi_centered < 0)
    # Total liquid volume
    total_liquid_volume = num_liquid_cells * cell_volume
    return total_liquid_volume
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Calculate the interface position (3D) from the fields files
#-------------------------------------------------------------------------------#
def calc_interface_position_3d(phi, hor1, hor2, vert):
    # Center around 0.0
    phi = phi - 0.5
    vert_sign_change = np.zeros((len(hor1), len(hor2)))
    for k in range(len(hor1)):
        for j in range(len(hor2)):
            for i in range(len(vert)):
                if phi[k, j, i] > 0.0:
                    vert_sign_change[k, j] = vert[i]
                    break
    
    hbar = np.mean(vert_sign_change)

    return hor1, hor2, vert_sign_change, hbar
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Calculate fft for 1d signal
#-------------------------------------------------------------------------------#
def fft_1d_signal(indep_variable, dep_variable):
    fft_y = np.fft.fft(dep_variable)
    fft_C = np.conjugate(fft_y)
    fft_magnitude = np.abs(fft_y*fft_C)
    frequencies = np.fft.fftfreq(len(indep_variable), 1/len(indep_variable))
    Np = len(indep_variable)
    Ener = np.zeros((Np))
    for col in range(Np):
        Ener[col] =  fft_magnitude[col]
    halfSpec = int(Np/2)
    Ek = Ener[1:halfSpec]/Np
    kvec_h = (frequencies[1:halfSpec])
    return kvec_h, Ek
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Calculate the dominant wavenumber and wavelength for the 1d signal
#-------------------------------------------------------------------------------#
def calc_dom_wavenumber_wavelength_1d(kvec_h, Ek, L):
    num_integrand_integL = np.multiply(Ek,np.reciprocal(kvec_h))
    num_integrand_domK = np.multiply(Ek,kvec_h)
    
    def mytrapz(xvar,yvar):
        svar = np.zeros(len(xvar))
        for i in range(0,len(xvar)-1):
            svar[i] =  (xvar[i+1]-xvar[i])*(yvar[i]+yvar[i+1])/2
        intgVar =  np.sum(svar)
        return intgVar
    numer_domK =  mytrapz(kvec_h,num_integrand_domK)
    denom = mytrapz(kvec_h,Ek)
    domK = numer_domK/denom
    domL = L/domK
    return domK, domL
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Calculate fft for 2d signal
#-------------------------------------------------------------------------------#
def fft_2d_signal(indep_variable1, indep_variable2, dep_variable):
    if dep_variable.shape != (len(indep_variable1), len(indep_variable2)):
        raise ValueError("Shape of dep_variable must match the sizes of indep_variable1 and indep_variable2.")
    
    dep_fft = np.fft.fft2(dep_variable)
    dep_fft_shifted = np.fft.fftshift(dep_fft)
    power_spectrum = np.abs(dep_fft_shifted)**2
    sample_period1 = abs(indep_variable1[1] - indep_variable1[0])
    sample_period2 = abs(indep_variable2[1] - indep_variable2[0])
    power_spectrum_normalized = power_spectrum / (len(indep_variable1) * len(indep_variable2))
    freq_h1 = np.fft.fftshift(np.fft.fftfreq(len(indep_variable1), sample_period1))
    freq_h2 = np.fft.fftshift(np.fft.fftfreq(len(indep_variable2), sample_period2))
    wavenumber_h1 = np.abs(freq_h1)
    wavenumber_h2 = np.abs(freq_h2)
    half_spec_h1 = len(indep_variable1) // 2
    half_spec_h2 = len(indep_variable2) // 2

    power_first_quadrant = power_spectrum_normalized[half_spec_h1:, half_spec_h2:]
    kvec_h1 = wavenumber_h1[half_spec_h1:]
    kvec_h2 = wavenumber_h2[half_spec_h2:]

    h1_profile = power_first_quadrant[0, :]
    h2_profile = power_first_quadrant[:, 0]
    power_first_quadrant[0, :] = 0.0
    power_first_quadrant[:, 0] = 0.0

    #return kvec_h1, h1_profile, kvec_h2, h2_profile
    return power_first_quadrant, kvec_h1, kvec_h2
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Calculate the dominant wavenumber and wavelength for the 2d signal
#  (works best for periodic boundary conditions and when the domain is equal in
#   both directions)
#-------------------------------------------------------------------------------#
def calc_dom_wavenumber_wavelength_2d(power_spectrum, kvec_h1, kvec_h2,Lx,Ly):
    max_idx = np.unravel_index(np.argmax(power_spectrum, axis=None), power_spectrum.shape)
    dom_kvec_h1 = kvec_h1[max_idx[0]]
    dom_kvec_h2 = kvec_h2[max_idx[1]]
    dom_WN_global = np.sqrt((dom_kvec_h1/Lx)**2 + (dom_kvec_h2/Ly)**2)
    dom_WL_global = 1.0/dom_WN_global
    return dom_WN_global, dom_WL_global
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Calculate the Flux for temperature
#-------------------------------------------------------------------------------#
def calc_Tflux(T, dx):
    Tflux = T[:,:,1] - T[:,:,0]
    Tflux = Tflux/dx
    return Tflux
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
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#
# Calculate the heat flux from the means file
#-------------------------------------------------------------------------------#
def calc_Nu_means(Tbar,hbar,xmr):
    # Obtain list of keys
    keys = list(Tbar.keys())
    result = np.zeros(len(keys))
    for key in keys:
        #result[keys.index(key)] = ((1.0 - Tbar[key][0])/xmr[0])* hbar[keys.index(key)]
        result[keys.index(key)] = ((Tbar[key][0] - Tbar[key][1])/(xmr[1] - xmr[0]))* hbar[keys.index(key)]
    return result