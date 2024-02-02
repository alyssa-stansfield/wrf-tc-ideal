import numpy as np
import matplotlib.pyplot as plt

def radprof_smooth(y, box_pts):
    #simple smoother for 1D arrays
    #box_pts = size of smoothing window
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def find_tc_radii(v_radprof,radprof_bins):
    #find radii of inner and outer regions of TC using radially-averaged profile of wind speed
    
    radprof_bins_centers = (radprof_bins[:-1] + radprof_bins[1:]) / 2 #bin centers

    v_radprof = np.nan_to_num(v_radprof)
    
    r_inner_ind = int(np.argmax(v_radprof)*2) #outer edge of inner-core = 2*radius of maximum wind
    r_inner_idx_grid = radprof_bins_centers[r_inner_ind] #index of r_inner

    idx_lowwind = np.where(v_radprof[r_inner_ind:]<8.0) #outer edge of TC = radius of 8 m/s wind speed
    if np.size(idx_lowwind) == 0: #if the wind profile never goes below 8 m/s
        r8_idx = (np.abs(v_radprof[r_inner_ind:] - 8.0)).argmin() #find wind speed closest to 8 m/s
    else:
        r8_idx = idx_lowwind[0][0]+r_inner_ind
    r_outer_idx_grid = radprof_bins_centers[r8_idx]
    
    #return the index value in radial profile of inner and outer radii, and bin center values
    return r_inner_idx_grid, r_outer_idx_grid, radprof_bins_centers

def find_tc_radii_precip(precip_radprof,radprof_bins,low_threshold):
    #find radii of inner and outer regions of TC using radially-averaged profile of accumulated precipitation
    #must define threshold of low precip to define outer radius of TC

    radprof_bins_centers = (radprof_bins[:-1] + radprof_bins[1:]) / 2 #bin centers

    precip_radprof = np.nan_to_num(precip_radprof)

    #first smooth the precipitation radial profile and then try to find first local minima
    precip_radprof_smooth = radprof_smooth(precip_radprof,5)
    r_bins_smooth = radprof_smooth(radprof_bins_centers[:-1],5)
    r_inner_ind = int(np.argmax(precip_radprof_smooth)*2) #outer edge of inner-core = 2*radius of maximum precip
    if r_inner_ind > len(r_bins_smooth):
        r_inner_ind = int(np.argmax(precip_radprof_smooth))
    r_inner_idx_grid = r_bins_smooth[r_inner_ind] #index of r_inner
    
    #use threshold of low precipitation amount (depends on units and accumulation length) to find outer radius of TC
    low_rain_idx = np.where(precip_radprof_smooth[r_inner_ind:]<low_threshold)
    r_outer_idx = low_rain_idx[0][0]+r_inner_ind
    r_outer_idx_grid = radprof_bins_centers[r_outer_idx]

    #return the index value in radial profile of inner and outer radii, and bin center values
    return r_inner_idx_grid, r_outer_idx_grid, radprof_bins_centers

def mean_inner_outer(r_inner,r_outer,var,r):
    #find mean of a 2D (x-y) variable in the inner-core of the TC and outer region, as defined by inner and outer radii as input. r is the distances from the TC center to each point in the domain
    var[var==0] = np.nan #replace 0s with NaNs so they aren't included in the means

    var_copy_inner = np.copy(var)
    var_copy_outer = np.copy(var)
    var_copy_inner[r>r_inner] = np.nan
    inner_mean = np.nanmean(var_copy_inner)

    var_copy_outer[r<=r_inner] = np.nan
    var_copy_outer[r>r_outer] = np.nan
    outer_mean = np.nanmean(var_copy_outer)

    return inner_mean, outer_mean

def mean_inner_outer_3D(r_inner,r_outer,var,r):
    #find mean of a 3D (height, x, y) variable in the inner-core of the TC and outer region, as defined by inner and outer radii as input. r is the distances from the TC center to each point in the domain
    var[var==0] = np.nan #replace 0s with NaNs so they aren't included in the means

    var_copy_inner = np.copy(var)
    var_copy_outer = np.copy(var)
    var_copy_inner[:,r>r_inner] = np.nan
    inner_mean = np.nanmean(var_copy_inner)

    var_copy_outer[:,r<=r_inner] = np.nan
    var_copy_outer[:,r>r_outer] = np.nan
    outer_mean = np.nanmean(var_copy_outer)

    return inner_mean, outer_mean

def max_inner_outer(r_inner,r_outer,var,r):
    #find maximum of a 2D (x-y) variable in the inner-core of the TC and outer region, as defined by inner and outer radii as input. r is the distances from the TC center to each point in the domain
    var[var==0] = np.nan #replace 0s with NaNs so they aren't included in the means

    var_copy_inner = np.copy(var)
    var_copy_outer = np.copy(var)
    var_copy_inner[r>r_inner] = np.nan
    inner_max = np.nanmax(var_copy_inner)

    var_copy_outer[r<=r_inner] = np.nan
    var_copy_outer[r>r_outer] = np.nan
    outer_max = np.nanmax(var_copy_outer)

    return inner_max, outer_max

def min_inner_outer(r_inner,r_outer,var,r):
    #find maximum of a 2D (x-y) variable in the inner-core of the TC and outer region, as defined by inner and outer radii as input. r is the distances from the TC center to each point in the domain
    var[var==0] = np.nan #replace 0s with NaNs so they aren't included in the means

    var_copy_inner = np.copy(var)
    var_copy_outer = np.copy(var)
    var_copy_inner[r>r_inner] = np.nan
    inner_min = np.nanmin(var_copy_inner)

    var_copy_outer[r<=r_inner] = np.nan
    var_copy_outer[r>r_outer] = np.nan
    outer_min = np.nanmin(var_copy_outer)

    return inner_min, outer_min
    
def count_inner_outer(r_inner,r_outer,value,var,r):
    #find count of a value in the inner-core of the TC and outer region, as defined by inner and outer radii as input. r is the distances from the TC center to each point in the domain. value is the value you are counting and var is the 2D variable containing the value.

    var_copy_inner = np.copy(var)
    var_copy_outer = np.copy(var)

    var_copy_inner[r>r_inner] = np.nan
    inner_count = np.count_nonzero(var_copy_inner == value)

    var_copy_outer[r<=r_inner] = np.nan
    var_copy_outer[r>r_outer] = np.nan
    outer_count = np.count_nonzero(var_copy_outer == value)

    return inner_count, outer_count


def count_inner_outer_totals(r_inner,r_outer,var,r):
    #find count gridpoints in inner core and outer regions, input any 2D variable

    var_copy_inner = np.ones_like(var)
    var_copy_outer = np.ones_like(var)

    var_copy_inner[r>r_inner] = np.nan
    inner_count = np.count_nonzero(var_copy_inner == 1)

    var_copy_outer[r<=r_inner] = np.nan
    var_copy_outer[r>r_outer] = np.nan
    outer_count = np.count_nonzero(var_copy_outer == 1)

    return inner_count, outer_count




def calculate_CFAD(data_x, data_y, bin_x, bin_y):
    #function for calculating CFADs from Hungjui
    
    data_x_flat = data_x.flatten()
    data_y_flat = data_y.flatten()
    
    print (np.shape(data_x_flat), np.shape(data_y_flat), np.shape(bin_x), np.shape(bin_y))

    data_cfad = plt.hist2d(data_x_flat, data_y_flat, bins=[bin_x, bin_y])
    plt.close()
    
    return data_cfad[0]

def updraft_counter(r_inner,r_outer,w_max_vals,r,w_threshold):
    #count number of updrafts (where updraft speed defined by w_threshold) in inner core and outer region
    w_max_vals[w_max_vals<w_threshold] = np.nan

    var_copy_inner = np.copy(w_max_vals)
    var_copy_outer = np.copy(w_max_vals)
    var_copy_inner[r>r_inner] = np.nan
    inner_count = np.count_nonzero(~np.isnan(var_copy_inner))

    var_copy_outer[r<=r_inner] = np.nan
    var_copy_outer[r>r_outer] = np.nan
    outer_count = np.count_nonzero(~np.isnan(var_copy_outer))

    return inner_count, outer_count
