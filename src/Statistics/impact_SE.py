import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.integrate import trapz
from scipy.stats import gaussian_kde
import warnings


def impact(data, cls, plot_it=False, pde=True, mean_lines=False, median_lines=False):
    unique_cls = np.unique(cls)
    if len(unique_cls) != 2:
        raise ValueError("Cls must contain exactly two distinct classes!")
    if len(data) != len(cls):
        raise ValueError("Data and Cls must have the same length!")
    if np.isnan(data).any():
        warnings.warn("NAs detected and removed.")
        valid_indices = ~np.isnan(data)
        cls = cls[valid_indices]
        data = data[valid_indices]

    dir_ct = 1
    dir_morph = 1
    impact_x2x1 = np.nan
    morph_diff = np.nan
    ct_diff = np.nan
    gmd_data = np.nan

    ks_pvalue = ks_2samp(data[cls == unique_cls[0]], data[cls == unique_cls[1]]).pvalue

    # if np.var(data) == 0 or ks_pvalue >= 0.05:
    #     impact_x2x1 = 0
    # else:
    median_cls1 = np.median(data[cls == unique_cls[0]])
    median_cls2 = np.median(data[cls == unique_cls[1]])
    delta_m = median_cls2 - median_cls1
    dir_ct = -1 if median_cls2 < median_cls1 else 1
    gmd_data = class_gmd(data, cls)
    ct_diff = abs(delta_m) / gmd_data
    ct_diff_weight = min(ct_diff, 2) / 2

    if pde:
        if (np.var(data[cls == unique_cls[0]]) == 0 or np.var(data[cls == unique_cls[1]]) == 0) and np.var(data) > 0:
            morph_diff = 0
        else:
            # Assuming ParetoDensityEstimationIE is already implemented and returns a dictionary
            # with keys 'kernels' and 'paretoDensity' for the density estimations.
            pde_results = pareto_density_estimation_ie(data)
            pde_x1 = pareto_density_estimation_ie(data[cls == unique_cls[0]], pde_results['kernels'])
            pde_x2 = pareto_density_estimation_ie(data[cls == unique_cls[1]], pde_results['kernels'])

            pde_diff = np.abs(pde_x2['paretoDensity'] - pde_x1['paretoDensity'])
            momentum1 = np.sum(np.sign(data[cls == unique_cls[0]]) * np.log10(np.abs(data[cls == unique_cls[0]]) + 1)) / \
                (np.sign(len(data[cls == unique_cls[0]])) * np.log10(np.abs(len(data[cls == unique_cls[0]])) + 1))
            momentum2 = np.sum(np.sign(data[cls == unique_cls[1]]) * np.log10(np.abs(data[cls == unique_cls[1]]) + 1)) / \
                (np.sign(len(data[cls == unique_cls[1]])) * np.log10(np.abs(len(data[cls == unique_cls[1]])) + 1))
            dir_morph = -1 if momentum2 < momentum1 else 1

            if len(pde_results['kernels']) == len(pde_diff):
                morph_diff = trapz(pde_diff, pde_results['kernels'])

    impact_x2x1 = ct_diff_weight * (dir_ct * abs(ct_diff)) + (1 - ct_diff_weight) * (dir_morph * abs(morph_diff))
    
    return {'Impact': impact_x2x1, 'MorphDiff': morph_diff, 'CTDiff': ct_diff}

def c_gmd(x):
    n = len(x)
    x_sorted = np.sort(x)
    return (2.0 / n**2) * sum((2 * i - n - 1) * x_sorted[i - 1] for i in range(1, n + 1))

def class_gmd(data, cls):
    unique_cls = np.unique(cls)
    if np.var(data) == 0:
        gmdn = 1e-7
    elif (np.var(data[cls == unique_cls[0]]) == 0 or np.var(data[cls == unique_cls[1]]) == 0) and np.var(data) > 0:
        gmdn = c_gmd(data)
    else:
        gmd1 = c_gmd(data[cls == unique_cls[0]])
        gmd2 = c_gmd(data[cls == unique_cls[1]])
        gmdn = np.sqrt((gmd1**2 + gmd2**2) / 2)
    return gmdn

def pareto_density_estimation_ie(data, kernels=None, min_anz_kernels=100):
    if len(np.unique(data)) < 3:
        raise ValueError("Data must contain more than 2 unique values for density estimation.")
    
    if kernels is None:
        # Optimal number of bins estimation and kernel placement
        n_bins = optimal_no_bins_ie(data)
        n_bins = max(min_anz_kernels, n_bins)  # Ensure at least min_anz_kernels
        kernels = np.linspace(np.min(data), np.max(data), n_bins)

    # Pareto density estimation
    pareto_radius = pareto_radius_ie(data)
    pareto_density = np.zeros_like(kernels)
    
    for i, kernel in enumerate(kernels):
        lb = kernel - pareto_radius
        ub = kernel + pareto_radius
        isInParetoSphere = np.logical_and(data >= lb, data <= ub)
        pareto_density[i] = np.sum(isInParetoSphere)
    
    # Normalize the density
    pareto_density /= np.trapz(pareto_density, kernels)
    
    return {'kernels': kernels, 'paretoDensity': pareto_density, 'paretoRadius': pareto_radius}

def optimal_no_bins_ie(data):
    # Freedman-Diaconis number of bins estimator
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr * len(data) ** (-1/3)
    return np.ceil((np.max(data) - np.min(data)) / bin_width).astype(int)

def pareto_radius_ie(data, maximum_nr_samples=10000):
    # Pareto radius estimation based on a sample of the data
    if maximum_nr_samples < len(data):
        sample_indices = np.random.choice(len(data), maximum_nr_samples, replace=False)
        sample_data = data[sample_indices]
    else:
        sample_data = data
    
    # Calculate the distances
    distances = np.abs(sample_data[:, None] - sample_data[None, :])
    pareto_radius = np.percentile(distances, 18)
    
    if pareto_radius == 0:
        pareto_radius = np.min(distances[distances > 0])
    
    if len(data) > 1024:
        pareto_radius *= 4 / (len(data)**0.2)
    
    return pareto_radius


# test code
# import matplotlib.pyplot as plt
# # Now, create a simple test example
# np.random.seed(42)  # For reproducibility

# # Generate two normally distributed samples with different means
# group1 = np.random.normal(loc=0, scale=1, size=100)
# group2 = np.random.normal(loc=0.5, scale=1, size=100)

# # Concatenate the groups into one data array and create a label array
# data = np.concatenate([group1, group2])
# cls = np.array([1]*100 + [2]*100)  # Labels: 1 for the first group, 2 for the second group

# # Apply the impact function
# result = impact(data, cls)

# print("Impact result:", result)

# # Optional: plot the distributions
# plt.hist(group1, bins=20, alpha=0.5, label='Group 1')
# plt.hist(group2, bins=20, alpha=0.5, label='Group 2')
# plt.legend()
# plt.show()
