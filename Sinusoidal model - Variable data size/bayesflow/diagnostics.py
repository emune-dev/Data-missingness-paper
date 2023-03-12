import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from scipy.stats import binom
import scipy.stats as stats
from scipy.integrate import solve_ivp, quad
from sklearn.metrics import r2_score, confusion_matrix, mean_squared_error
#from matplotlib.ticker import FormatStrFormatter

from bayesflow.computational_utilities import expected_calibration_error


def true_vs_estimated(theta_true, theta_est, param_names, figsize=(8, 4), show=True, filename=None, font_size=12):
    """ Plots a scatter plot with abline of the estimated posterior means vs true values. """

    # Plot settings
    plt.rcParams['font.size'] = font_size

    # Determine n_subplots dynamically
    n_row = int(np.ceil(len(param_names) / 6))
    n_col = int(np.ceil(len(param_names) / n_row))

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat
        
    # --- Plot true vs estimated posterior means on a single row --- #
    for j in range(len(param_names)):
        
        # Plot analytic vs estimated
        axarr[j].scatter(theta_est[:, j], theta_true[:, j], color='black', alpha=0.4)
        
        # get axis limits and set equal x and y limits
        lower_lim = min(axarr[j].get_xlim()[0], axarr[j].get_ylim()[0])
        upper_lim = max(axarr[j].get_xlim()[1], axarr[j].get_ylim()[1])
        axarr[j].set_xlim((lower_lim, upper_lim))
        axarr[j].set_ylim((lower_lim, upper_lim))
        axarr[j].plot(axarr[j].get_xlim(), axarr[j].get_xlim(), '--', color='black')
        
        # Compute NRMSE
        rmse = np.sqrt(np.mean( (theta_est[:, j] - theta_true[:, j])**2 ))
        nrmse = rmse / (theta_true[:, j].max() - theta_true[:, j].min())
        axarr[j].text(0.1, 0.9, 'NRMSE={:.3f}'.format(nrmse),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes,
                     size=12)
        
        # Compute R2
        r2 = r2_score(theta_true[:, j], theta_est[:, j])
        axarr[j].text(0.1, 0.8, '$R^2$={:.3f}'.format(r2),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes, 
                     size=12)
        
        if j == 0:
            # Label plot
            axarr[j].set_xlabel('Estimated')
            axarr[j].set_ylabel('True')
        axarr[j].set_title(param_names[j])
        axarr[j].spines['right'].set_visible(False)
        axarr[j].spines['top'].set_visible(False)
    
    # Adjust spaces
    f.tight_layout()
    if show:
        plt.show()
    # Save if specified
    if filename is not None:
        f.savefig("figures/{}_metrics.png".format(filename), dpi=300, bbox_inches='tight')
    return f


def plot_sbc(theta_samples, theta_test, param_names, bins=25, figsize=(8, 4), interval=0.99, show=True, filename=None, font_size=12):
    """ Plots the simulation-based posterior checking histograms as advocated by Talts et al. (2018). """

    # Plot settings
    plt.rcParams['font.size'] = font_size
    N = int(theta_test.shape[0])

    # Determine n_subplots dynamically
    n_row = int(np.ceil(len(param_names) / 6))
    n_col = int(np.ceil(len(param_names) / n_row))

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat

    # Compute ranks (using broadcasting)    
    ranks = np.sum(theta_samples < theta_test[:, np.newaxis, :], axis=1)
    
    # Compute interval
    endpoints = binom.interval(interval, N, 1 / (bins))

    # Plot histograms
    for j in range(len(param_names)):
        
        # Add interval
        axarr[j].axhspan(endpoints[0], endpoints[1], facecolor='gray', alpha=0.3)
        axarr[j].axhline(np.mean(endpoints), color='gray', zorder=0, alpha=0.5)
        
        sns.histplot(ranks[:, j], kde=False, ax=axarr[j], color='#a34f4f', bins=bins, alpha=0.95)
        
        axarr[j].set_title(param_names[j])
        axarr[j].spines['right'].set_visible(False)
        axarr[j].spines['top'].set_visible(False)
        if j == 0:
            axarr[j].set_xlabel('Rank statistic')
        axarr[j].get_yaxis().set_ticks([])
        axarr[j].set_ylabel('')
    
    f.tight_layout()
    # Show, if specified
    if show:
        plt.show()
    # Save if specified
    if filename is not None:
        f.savefig("figures/{}_SBC.png".format(filename), dpi=300, bbox_inches='tight')
    return f


def plot_2D(param_samples, posterior_xy, show_level_set=True, param_prior=None, filename=None):
    fig = plt.figure(figsize=(10, 10))
    # Level sets of analytic posterior distribution
    if show_level_set is True:
        mean_sample = np.mean(param_samples, axis=0)
        cov_sample = np.cov(param_samples.transpose())
        grid = 201
        A = np.linspace(mean_sample[0] - 3 * np.sqrt(cov_sample[0, 0]), mean_sample[0] + 3 * np.sqrt(cov_sample[0, 0]),
                        grid)
        B = np.linspace(mean_sample[1] - 3 * np.sqrt(cov_sample[1, 1]), mean_sample[1] + 3 * np.sqrt(cov_sample[1, 1]),
                        grid)
        true_posterior = np.zeros((grid, grid))
        for iy in range(0, grid):
            for ix in range(0, grid):
                true_posterior[iy][ix] = posterior_xy(A[ix], B[iy])
        true_posterior = plt.contour(A, B, true_posterior, colors='blue')
        h1,_ = true_posterior.legend_elements()
        #plt.clabel(true_posterior, fontsize=12, inline=1)

    # Kernel density estimator of BayesFlow samples
    a = param_samples[:, 0]
    b = param_samples[:, 1]
    ab = np.vstack([a, b])
    z = stats.gaussian_kde(ab)(ab)
    ida = z.argsort()  # Sort the points by density, so that the densest points are plotted last
    a, b, z = a[ida], b[ida], z[ida]
    approximate_posterior = plt.scatter(a, b, c=z, s=50)
    h2,_ = approximate_posterior.legend_elements()
    plt.legend([h2[0], h1[0]], ['BayesFlow samples', 'True posterior'])

    # True prior parameter
    if param_prior is not None:
        plt.scatter(param_prior[0, 0], param_prior[0, 1], color="red", marker="x", s=150)
    
    plt.xlabel('Parameter $k_1$')
    plt.ylabel('Parameter $k_2$')
    plt.show()

    # Save if specified
    if filename is not None:
        fig.savefig("figures/{}_2D_plot.png".format(filename), dpi=600, bbox_inches='tight')


def plot_marginal(param_samples, posterior_xy, filename=None):
    mean_sample = np.mean(param_samples, axis=0)
    cov_sample = np.cov(param_samples.transpose())
    mean_x = mean_sample[0]
    mean_y = mean_sample[1]
    std_x = np.sqrt(cov_sample[0, 0])
    std_y = np.sqrt(cov_sample[1, 1])

    grid = 201
    A = np.linspace(mean_x - 5 * std_x, mean_x + 5 * std_x, grid)
    B = np.linspace(mean_y - 5 * std_y, mean_y + 5 * std_y, grid)
    bounds = np.array([mean_x-8*std_x, mean_x+8*std_x, mean_y-8*std_y, mean_y+8*std_y])

    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(param_samples[:, 0], bins='auto', density=1, color='orange', label='BayesFlow samples')
    marginal_x = np.zeros(grid)
    for i in range(grid):
        x = A[i]
        integrand_y = lambda y: posterior_xy(x, y)
        marginal_x[i] = quad(integrand_y, bounds[2], bounds[3])[0]
    plt.plot(A, marginal_x, color='b', label='True posterior')
    plt.ylabel('Marginal density')
    plt.xlabel('Parameter $k_1$')
    plt.tight_layout()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(param_samples[:, 1], bins='auto', density=1, color='orange', label='BayesFlow samples')
    marginal_y = np.zeros(grid)
    for j in range(grid):
        y = B[j]
        integrand_x = lambda x: posterior_xy(x, y)
        marginal_y[j] = quad(integrand_x, bounds[0], bounds[1])[0]
    plt.plot(B, marginal_y, color='b', label='True posterior')
    plt.ylabel('Marginal density')
    plt.xlabel('Parameter $k_2$')
    plt.tight_layout()
    plt.legend()
    plt.show()

    # Save if specified
    if filename is not None:
        fig.savefig("figures/{}_marginal.png".format(filename), dpi=600, bbox_inches='tight')
        
        
def plot_posterior(param_samples, posterior_xy, filename=None, font_size=12):
    mean_sample = np.mean(param_samples, axis=0)
    cov_sample = np.cov(param_samples.transpose())
    mean_x = mean_sample[0]
    mean_y = mean_sample[1]
    std_x = np.sqrt(cov_sample[0, 0])
    std_y = np.sqrt(cov_sample[1, 1])

    fig = plt.figure(figsize=(15, 5))
    plt.rcParams['font.size'] = font_size
    
    plt.subplot(1, 3, 1)
    # Level sets of analytic posterior distribution
    grid = 201
    A = np.linspace(mean_x - 3 * std_x, mean_x + 3 * std_x, grid)
    B = np.linspace(mean_y - 3 * std_y, mean_y + 3 * std_y, grid)
    true_posterior = np.zeros((grid, grid))
    for iy in range(0, grid):
        for ix in range(0, grid):
            true_posterior[iy][ix] = posterior_xy(A[ix], B[iy])
    true_posterior = plt.contour(A, B, true_posterior, levels=None, colors='blue')
    h1, _ = true_posterior.legend_elements()
    #plt.clabel(true_posterior, fontsize=4, inline=1)
    # Kernel density estimator of BayesFlow samples
    a = param_samples[:, 0]
    b = param_samples[:, 1]
    ab = np.vstack([a, b])
    z = stats.gaussian_kde(ab)(ab)
    ida = z.argsort()  # Sort the points by density, so that the densest points are plotted last
    a, b, z = a[ida], b[ida], z[ida]
    approximate_posterior = plt.scatter(a, b, c=z, s=30)
    h2, _ = approximate_posterior.legend_elements()
    plt.legend([h2[0], h1[0]], ['BayesFlow', 'True posterior'], fontsize=11.5)
    plt.xlabel('Parameter $a$', fontsize=14)
    plt.ylabel('Parameter $b$', fontsize=14)

    # Check marginal densities
    grid = 151
    A = np.linspace(mean_x - 5 * std_x, mean_x + 5 * std_x, grid)
    B = np.linspace(mean_y - 5 * std_y, mean_y + 5 * std_y, grid)
    bounds = np.array([mean_x - 8 * std_x, mean_x + 8 * std_x, mean_y - 8 * std_y, mean_y + 8 * std_y])
    plt.subplot(1, 3, 2)
    plt.hist(param_samples[:, 0], bins='auto', density=1, color='orange', label='BayesFlow')
    marginal_x = np.zeros(grid)
    for i in range(grid):
        x = A[i]
        integrand_y = lambda y: posterior_xy(x, y)
        marginal_x[i] = quad(integrand_y, bounds[2], bounds[3])[0]
    plt.plot(A, marginal_x, color='b', label='True posterior')
    plt.ylabel('Marginal density', fontsize=14)
    plt.xlabel('Parameter $a$', fontsize=14)
    plt.legend(fontsize=11.5)
    plt.subplot(1, 3, 3)
    plt.hist(param_samples[:, 1], bins='auto', density=1, color='orange', label='BayesFlow')
    marginal_y = np.zeros(grid)
    for j in range(grid):
        y = B[j]
        integrand_x = lambda x: posterior_xy(x, y)
        marginal_y[j] = quad(integrand_x, bounds[0], bounds[1])[0]
    plt.plot(B, marginal_y, color='b', label='True posterior')
    plt.ylabel('Marginal density', fontsize=14)
    plt.xlabel('Parameter $b$', fontsize=14)
    plt.legend(fontsize=11.5)

    plt.tight_layout()
    plt.show()
    # Save if specified
    if filename is not None:
        fig.savefig("figures/{}_posterior.png".format(filename), dpi=300, bbox_inches='tight')
    
    
def plot_posterior_comparison(param_samples, A, B, C, D, true_posterior, marginal_x, marginal_y, row, method, 
                    fontsize=13, levels=None, label_1=None, label_2=None): 
    plt.subplot(3, 3, 3*row-2) 
    if levels is not None:
        true_posterior = plt.contour(A, B, true_posterior, levels, colors='blue')
    else:
        true_posterior = plt.contour(A, B, true_posterior, colors='blue')
        plt.clabel(true_posterior, fontsize=6, inline=1) ###
    h1, _ = true_posterior.legend_elements()
    
    # Kernel density estimator of BayesFlow samples
    a = param_samples[:, 0]
    b = param_samples[:, 1]
    ab = np.vstack([a, b])
    z = stats.gaussian_kde(ab)(ab)
    ida = z.argsort()  # Sort the points by density, so that the densest points are plotted last
    a, b, z = a[ida], b[ida], z[ida]   
    plt.xticks([-0.90, -0.85, -0.80, -0.75])
    #plt.xticks([-0.48, -0.46, -0.44, -0.42, -0.40, -0.38, -0.36])
    #plt.yticks([-1.0, -0.8, -0.6, -0.4, -0.2])
    approximate_posterior = plt.scatter(a, b, c=z, s=30)
    h2, _ = approximate_posterior.legend_elements()
    plt.title(method, fontsize=25.5, loc='left', pad=8)
    plt.legend([h2[0], h1[0]], [label_2, label_1], fontsize=14)
    plt.xlabel('Parameter $k_1$', fontsize=20)
    plt.ylabel('Parameter $k_2$', fontsize=20)

    # Check marginal densities    
    plt.subplot(3, 3, 3*row-1)
    plt.xticks([-0.90, -0.85, -0.80, -0.75])
    plt.yticks([0,5,10,15,20])
    plt.hist(param_samples[:, 0], bins='auto', density=1, color='orange', label=label_2) 
    plt.plot(C, marginal_x, color='b', label=label_1)
    plt.ylabel('Marginal density', fontsize=20)
    plt.xlabel('Parameter $k_1$', fontsize=20)
    plt.legend(fontsize=fontsize, loc='upper left')
    
    plt.subplot(3, 3, 3*row)
    #plt.xticks([-1.75, -1.25, -0.75, -0.25, 0.25])
    #plt.yticks([0,1,2,3,4])
    plt.hist(param_samples[:, 1], bins='auto', density=1, color='orange', label=label_2)    
    plt.plot(D, marginal_y, color='b', label=label_1)
    plt.ylabel('Marginal density', fontsize=20)
    plt.xlabel('Parameter $k_2$', fontsize=20)
    plt.legend(fontsize=fontsize, loc='upper left')
    
    
def plot_2D_comparison(param_samples, A, B, true_posterior, column, method):     
    plt.subplot(1, 3, column)  
    levels = np.array([10, 25, 47, 70, 90])
    true_posterior = plt.contour(A, B, true_posterior, levels, colors='blue')
    h1, _ = true_posterior.legend_elements()
    
    # Kernel density estimator of BayesFlow samples
    a = param_samples[:, 0]
    b = param_samples[:, 1]
    ab = np.vstack([a, b])
    z = stats.gaussian_kde(ab)(ab)
    ida = z.argsort()  # Sort the points by density, so that the densest points are plotted last
    a, b, z = a[ida], b[ida], z[ida]
    #plt.xticks([-0.90, -0.85, -0.80, -0.75])
    plt.yticks([-1.0, -0.8, -0.6, -0.4, -0.2])
    #plt.xlim(left = -0.493)
    #plt.ylim(-1.9, 0.4)
    approximate_posterior = plt.scatter(a, b, c=z, s=30)
    h2, _ = approximate_posterior.legend_elements()
    plt.title(method, fontsize=20, pad=8)
    plt.legend([h2[0], h1[0]], ['BayesFlow', 'True posterior'], fontsize=11)
    plt.xlabel('Parameter $k_1$', fontsize=14)
    plt.ylabel('Parameter $k_2$', fontsize=14)    

    
def plot_predictive_2D(ax, param_prior, result, param_samples, A, B, true_posterior, method=None, n_sim=301, linewidth=0.6):
    def conversion_reaction(t, x, theta):
        theta = 10**theta
        return np.array([-theta[0]*x[0]+theta[1]*x[1], theta[0]*x[0]-theta[1]*x[1]])
    x0 = [1,0]      
    n_obs = 11
    time_points = np.linspace(0, 10, n_obs)
    
    # Posterior predictive check
    for k in range(n_sim):
        rhs = lambda t,x: conversion_reaction(t, x, param_samples[k])
        sol = solve_ivp(rhs, t_span = (0,10), y0 = x0, atol = 1e-9, rtol = 1e-6)
        if k == 0: 
            ax[0].plot(sol.t[0], sol.y[1][0], color='grey', label='Simulation', linewidth=0.9)
        else: 
            ax[0].plot(sol.t, sol.y[1], color='grey', linewidth=linewidth, alpha=0.3) #0.3
    rhs = lambda t,x: conversion_reaction(t, x, param_prior[0])
    sol = solve_ivp(rhs, t_span = (0,10), y0 = x0, atol = 1e-9, rtol = 1e-6)
    ax[0].plot(sol.t, sol.y[1], color='black', label='True trajectory') 
    present_indices = result[1]
    missing_indices = np.setdiff1d(range(n_obs), present_indices)
    ax[0].plot(time_points[present_indices], result[0][present_indices], 'o', color='blue', label='Available data')
    ax[0].set_xlabel('Time $t$', fontsize=15)
    ax[0].set_ylabel('Measurement $y$', fontsize=15)    
    if method is not None:
        ax[0].set_title(method, fontsize=20, loc='left', pad=8)    
    handles, labels = ax[0].get_legend_handles_labels()
    order = [1,2,0]
    ax[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=13)
    
    # 2D plot
    true_posterior = ax[1].contour(A, B, true_posterior, levels=np.array([5, 13.5, 26, 38.5, 51]), colors='blue')
    #plt.clabel(true_posterior, fontsize=9, inline=1)
    h1, _ = true_posterior.legend_elements()
    # Kernel density estimator of BayesFlow samples
    a = param_samples[:, 0]
    b = param_samples[:, 1]
    ab = np.vstack([a, b])
    z = stats.gaussian_kde(ab)(ab)
    ida = z.argsort()  # Sort the points by density, so that the densest points are plotted last
    a, b, z = a[ida], b[ida], z[ida]
    approximate_posterior = ax[1].scatter(a, b, c=z, s=30)
    h2, _ = approximate_posterior.legend_elements()
    ax[1].legend([h2[0], h1[0]], ['BayesFlow', 'True posterior'], fontsize=13)
    ax[1].set_xlabel('Parameter $k_1$', fontsize=15)
    ax[1].set_ylabel('Parameter $k_2$', fontsize=15)    
    

def error_metrics(true_params, param_samples):
    v = np.empty(5)    
    param_means = param_samples.mean(axis=1)
    for j in range(2):
        # Compute NRMSE
        rmse = np.sqrt(np.mean( (param_means[:, j] - true_params[:, j])**2 ))
        nrmse = rmse / (true_params[:, j].max() - true_params[:, j].min())
        v[2*j] = nrmse        
        # Compute R²
        r2 = r2_score(true_params[:, j], param_means[:, j])
        v[2*j+1] = r2
    
    # Compute other MSE (as proposed by Yannik)
    a = []
    for m in range(500):
        mse = mean_squared_error(np.ones((1000,1))@np.array([true_params[m]]), param_samples[m], squared=False)
        a.append(mse)  
    v[4] = np.mean(np.array(a))    
    return v