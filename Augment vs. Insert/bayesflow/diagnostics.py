import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from scipy.stats import binom, norm
import scipy.stats as stats
from scipy.integrate import quad, solve_ivp, dblquad
from sklearn.metrics import r2_score, confusion_matrix, mean_squared_error
#from matplotlib.ticker import FormatStrFormatter

from bayesflow.computational_utilities import expected_calibration_error


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
        f.savefig("figures/{}_SBC.png".format(filename), dpi=600, bbox_inches='tight')
    return f
     
        
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
    true_posterior = plt.contour(A, B, true_posterior, colors='blue')
    h1, _ = true_posterior.legend_elements()
    # plt.clabel(true_posterior, fontsize=4, inline=1)
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
    plt.xlabel('Parameter $k_1$', fontsize=14)
    plt.ylabel('Parameter $k_2$', fontsize=14)

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
    plt.xlabel('Parameter $k_1$', fontsize=14)
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
    plt.xlabel('Parameter $k_2$', fontsize=14)
    plt.legend(fontsize=11.5)

    plt.tight_layout()
    plt.show()
    # Save if specified
    if filename is not None:
        fig.savefig("figures/{}_posterior.png".format(filename), dpi=600, bbox_inches='tight')
    
    
def plot_predictive_2D(ax, param_prior, result, param_samples, A, B, true_posterior, method, n_sim=301, linewidth=0.6):
    def conversion_reaction(t, x, theta):
        theta = 10**theta
        return np.array([-theta[0]*x[0]+theta[1]*x[1], theta[0]*x[0]-theta[1]*x[1]])
    x0 = [1,0]      
    n_obs = 3
    time_points = np.linspace(0, 10, n_obs)
    
    # Posterior predictive check
    for k in range(n_sim):
        rhs = lambda t,x: conversion_reaction(t, x, param_samples[k])
        sol = solve_ivp(rhs, t_span = (0,10), y0 = x0, atol = 1e-9, rtol = 1e-6)
        if k == 0: 
            ax[0].plot(sol.t[0], sol.y[1][0], color='grey', label='Simulation', linewidth=0.9)
        else: 
            ax[0].plot(sol.t, sol.y[1], color='grey', linewidth=linewidth, alpha=0.32) #0.3
    ax[0].plot(np.linspace(0, 10, 21), 0.5*np.ones(21), '--', color='c', linewidth=1)
    rhs = lambda t,x: conversion_reaction(t, x, param_prior[0])
    sol = solve_ivp(rhs, t_span = (0,10), y0 = x0, atol = 1e-9, rtol = 1e-6)
    ax[0].plot(sol.t, sol.y[1], color='black', label='True trajectory') 
    present_indices = result[1]
    missing_indices = np.setdiff1d(range(n_obs), present_indices)
    ax[0].plot(time_points[present_indices], result[0][present_indices], 'o', color='blue', label='Available data')
    ax[0].plot(time_points[missing_indices], result[0][missing_indices], 'o', color='red', label='Missing data')
    ax[0].plot(time_points[missing_indices], 0.5*np.ones(len(missing_indices)), 'o', color='c', label='Fill-in value')
    ax[0].set_xlabel('Time $t$', fontsize=15)
    ax[0].set_ylabel('Measurement $y$', fontsize=15)    
    ax[0].set_title(method, fontsize=20, loc='left', pad=8)    
    handles, labels = ax[0].get_legend_handles_labels()
    order = [1,2,3,4,0]
    ax[0].legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=13)
    
    # 2D plot
    # levels = np.array([9, 23.75, 47.5, 71.25, 95])
    #levels = np.array([3, 10.5, 20, 29]) 
    true_posterior = ax[1].contour(A, B, true_posterior, colors='blue')
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
    

"""    
def plot_posterior(ax, col, result, param_samples, method, levels):  
    sigma = 0.015   # noise standard deviation
    
    def prior_eval(x,y):   
        # Evaluates prior probability p(theta) according to k_1, k_2 ~ N(-0.75, 0.25²) iid.
        return norm.pdf(x,-0.75,0.25) * norm.pdf(y,-0.75,0.25)

    def likelihood(x,y):   
        # Calculates likelihood p(x_{1:N} | theta) by ignoring the missing data
        x = 10**x
        y = 10**y
        s = x + y
        b = x/s
        state_2 = lambda t: b - b * np.exp(-s*t)
        sol = state_2(result[2])
        residual = (result[0][result[1]] - sol)/sigma
        nllh = np.sum(np.log(2*np.pi*sigma**2)+residual**2)/2
        return np.exp(-nllh)

    def unnormalized_posterior(x,y):   
        # Evaluates the unnormalized posterior probability p(theta | x_{1:N}) according to Bayes' formula
        return likelihood(x,y) * prior_eval(x,y)

    # scaling factor
    scaling_factor = dblquad(unnormalized_posterior, -2.25, 0.75, lambda y: -2.25, lambda y: 0.75)
    posterior_xy = lambda x,y: unnormalized_posterior(x,y)/scaling_factor[0]
    
    mean_sample = np.mean(param_samples, axis=0)
    cov_sample = np.cov(param_samples.transpose())
    mean_x = mean_sample[0]
    mean_y = mean_sample[1]
    std_x = np.sqrt(cov_sample[0, 0])
    std_y = np.sqrt(cov_sample[1, 1])

    # Level sets of analytic posterior distribution
    grid = 201
    A = np.linspace(mean_x - 3 * std_x, mean_x + 3 * std_x, grid)
    B = np.linspace(mean_y - 3 * std_y, mean_y + 3 * std_y, grid)
    true_posterior = np.zeros((grid, grid))
    for iy in range(0, grid):
        for ix in range(0, grid):
            true_posterior[iy][ix] = posterior_xy(A[ix], B[iy])      
    true_posterior = ax.contour(A, B, true_posterior, levels=levels, colors='blue')  
    #plt.clabel(true_posterior, fontsize=9, inline=1)
    h1, _ = true_posterior.legend_elements()
    
    # Kernel density estimator of BayesFlow samples
    a = param_samples[:, 0]
    b = param_samples[:, 1]
    ab = np.vstack([a, b])
    z = stats.gaussian_kde(ab)(ab)
    ida = z.argsort()  # Sort the points by density, so that the densest points are plotted last
    a, b, z = a[ida], b[ida], z[ida]
    approximate_posterior = ax.scatter(a, b, c=z, s=25)
    h2, _ = approximate_posterior.legend_elements()
    if col == 1:
        ax.legend([h2[0], h1[0]], ['BayesFlow', 'True posterior'], loc='lower center', bbox_to_anchor=(0.5, -0.38), ncol=2)
    ax.set_xlabel('Parameter $k_1$')
    ax.set_ylabel('Parameter $k_2$')    
    ax.set_title(method, fontsize=13.5, loc='center', pad=7) 
    

def predictive_check(ax, col, param_prior, result, param_samples, n_sim=301):
    def conversion_reaction(t, x, theta):
        theta = 10**theta
        return np.array([-theta[0]*x[0]+theta[1]*x[1], theta[0]*x[0]-theta[1]*x[1]])
    x0 = [1,0]      
    n_obs = 3
    time_points = np.linspace(0, 10, n_obs)
    
    # Posterior predictive check
    for k in range(n_sim):
        rhs = lambda t,x: conversion_reaction(t, x, param_samples[k])
        sol = solve_ivp(rhs, t_span = (0,10), y0 = x0, atol = 1e-9, rtol = 1e-6)
        if k == 0: 
            ax.plot(sol.t[0], sol.y[1][0], color='grey', label='Simulation', linewidth=0.9)
        else: 
            ax.plot(sol.t, sol.y[1], color='grey', linewidth=0.6, alpha=0.3) 
    ax.plot(np.linspace(0, 10, 21), 0.5*np.ones(21), '--', color='c', linewidth=1)
    rhs = lambda t,x: conversion_reaction(t, x, param_prior[0])
    sol = solve_ivp(rhs, t_span = (0,10), y0 = x0, atol = 1e-9, rtol = 1e-6)
    ax.plot(sol.t, sol.y[1], color='black', label='True trajectory') 
    present_indices = result[1]
    missing_indices = np.setdiff1d(range(n_obs), present_indices)
    ax.plot(time_points[present_indices], result[0][present_indices], 'o', color='blue', label='Available data')
    ax.plot(time_points[missing_indices], result[0][missing_indices], 'o', color='red', label='Missing data')
    ax.plot(time_points[missing_indices], 0.5*np.ones(len(missing_indices)), 'o', color='c', label='Fill-in value')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Measurement $y$')     
    if col == 1:
        handles, labels = ax.get_legend_handles_labels()
        order = [1,2,3,4,0]
        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=5)    
"""    
       
    
def plot_posterior(ax, j, result, param_samples, method, levels):  
    sigma = 0.015   # noise standard deviation
    
    def prior_eval(x,y):   
        # Evaluates prior probability p(theta) according to k_1, k_2 ~ N(-0.75, 0.25²) iid.
        return norm.pdf(x,-0.75,0.25) * norm.pdf(y,-0.75,0.25)

    def likelihood(x,y):   
        # Calculates likelihood p(x_{1:N} | theta) by ignoring the missing data
        x = 10**x
        y = 10**y
        s = x + y
        b = x/s
        state_2 = lambda t: b - b * np.exp(-s*t)
        sol = state_2(result[2])
        residual = (result[0][result[1]] - sol)/sigma
        nllh = np.sum(np.log(2*np.pi*sigma**2)+residual**2)/2
        return np.exp(-nllh)

    def unnormalized_posterior(x,y):   
        # Evaluates the unnormalized posterior probability p(theta | x_{1:N}) according to Bayes' formula
        return likelihood(x,y) * prior_eval(x,y)

    # scaling factor
    scaling_factor = dblquad(unnormalized_posterior, -2.25, 0.75, lambda y: -2.25, lambda y: 0.75)
    posterior_xy = lambda x,y: unnormalized_posterior(x,y)/scaling_factor[0]
    
    mean_sample = np.mean(param_samples, axis=0)
    cov_sample = np.cov(param_samples.transpose())
    mean_x = mean_sample[0]
    mean_y = mean_sample[1]
    std_x = np.sqrt(cov_sample[0, 0])
    std_y = np.sqrt(cov_sample[1, 1])

    # Level sets of analytic posterior distribution
    grid = 201
    A = np.linspace(mean_x - 3 * std_x, mean_x + 3 * std_x, grid)
    B = np.linspace(mean_y - 3 * std_y, mean_y + 3 * std_y, grid)
    true_posterior = np.zeros((grid, grid))
    for iy in range(0, grid):
        for ix in range(0, grid):
            true_posterior[iy][ix] = posterior_xy(A[ix], B[iy])      
    true_posterior = ax[0,j].contour(A, B, true_posterior, levels=levels, colors='blue')  
    #plt.clabel(true_posterior, fontsize=9, inline=1)
    h1, _ = true_posterior.legend_elements()
    
    # Kernel density estimator of BayesFlow samples
    a = param_samples[:, 0]
    b = param_samples[:, 1]
    ab = np.vstack([a, b])
    z = stats.gaussian_kde(ab)(ab)
    ida = z.argsort()  # Sort the points by density, so that the densest points are plotted last
    a, b, z = a[ida], b[ida], z[ida]
    approximate_posterior = ax[0,j].scatter(a, b, c=z, s=25)
    h2, _ = approximate_posterior.legend_elements()
    if j == 2:
        ax[0,j].legend([h2[0], h1[0]], ['BayesFlow', 'True posterior'], bbox_to_anchor=(1.04,0.61))
    ax[0,j].set_xlabel('Parameter $k_1$')
    ax[0,j].set_ylabel('Parameter $k_2$')    
    ax[0,j].set_title(method, fontsize=14, loc='center', pad=8.5) 
    

def predictive_check(ax, j, param_prior, result, param_samples, n_sim=301):
    def conversion_reaction(t, x, theta):
        theta = 10**theta
        return np.array([-theta[0]*x[0]+theta[1]*x[1], theta[0]*x[0]-theta[1]*x[1]])
    x0 = [1,0]      
    n_obs = 3
    time_points = np.linspace(0, 10, n_obs)
    
    # Posterior predictive check
    for k in range(n_sim):
        rhs = lambda t,x: conversion_reaction(t, x, param_samples[k])
        sol = solve_ivp(rhs, t_span = (0,10), y0 = x0, atol = 1e-9, rtol = 1e-6)
        if k == 0: 
            ax[1,j].plot(sol.t[0], sol.y[1][0], color='grey', label='Simulation', linewidth=0.9)
        else: 
            ax[1,j].plot(sol.t, sol.y[1], color='grey', linewidth=0.6, alpha=0.3) 
    ax[1,j].plot(np.linspace(0, 10, 21), 0.5*np.ones(21), '--', color='c', linewidth=1)
    rhs = lambda t,x: conversion_reaction(t, x, param_prior[0])
    sol = solve_ivp(rhs, t_span = (0,10), y0 = x0, atol = 1e-9, rtol = 1e-6)
    ax[1,j].plot(sol.t, sol.y[1], color='black', label='True trajectory') 
    present_indices = result[1]
    missing_indices = np.setdiff1d(range(n_obs), present_indices)
    ax[1,j].plot(time_points[present_indices], result[0][present_indices], 'o', color='blue', label='Available data')
    ax[1,j].plot(time_points[missing_indices], result[0][missing_indices], 'o', color='red', label='Missing data')
    ax[1,j].plot(time_points[missing_indices], 0.5*np.ones(len(missing_indices)), 'o', color='c', label='Fill-in value')
    ax[1,j].set_xlabel('Time $t$')
    ax[1,j].set_ylabel('Measurement $y$')     
    if j == 2:
        handles, labels = ax[1,j].get_legend_handles_labels()
        order = [1,2,3,4,0]
        ax[1,j].legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(1.04,0.74))