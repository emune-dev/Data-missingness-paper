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


def true_vs_estimated(theta_true, theta_est, param_names, dpi=300, figsize=(20, 4), show=True, filename=None, font_size=12):
    """ Plots a scatter plot with abline of the estimated posterior means vs true values.

    Parameters
    ----------
    theta_true: np.array
        Array of true parameters.
    theta_est: np.array
        Array of estimated parameters.
    param_names: list(str)
        List of parameter names for plotting.
    figsize: tuple(int, int), default: (20,4)
        Figure size.
    show: boolean, default: True
        Controls if the plot will be shown
    filename: str, default: None
        Filename if plot shall be saved
    font_size: int, default: 12
        Font size
    """


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
                     size=10)
        
        # Compute R2
        r2 = r2_score(theta_true[:, j], theta_est[:, j])
        axarr[j].text(0.1, 0.8, '$R^2$={:.3f}'.format(r2),
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform=axarr[j].transAxes, 
                     size=10)
        
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
        f.savefig("figures/{}_metrics.png".format(filename), dpi=600, bbox_inches='tight')
    return f


def plot_sbc(theta_samples, theta_test, param_names, bins=25, figsize=(24, 12), interval=0.99, show=True, filename=None, font_size=12):
    """ Plots the simulation-based posterior checking histograms as advocated by Talts et al. (2018).

    Parameters
    ----------
    theta_samples: np.array
        Array of sampled parameters
    theta_test: np.array
        Array of test parameters
    param_names: list(str)
        List of parameter names for plotting.
    bins: int, default: 25
        Bins for histogram plot
    figsize: tuple(int, int), default: (24, 12)
        Figure size
    interval: float, default: 0.99
        Interval to plot
    show: bool, default: True
        Controls whether the plot shall be printed
    font_size: int, default:12
        Font size

    """

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


def plot_confusion_matrix(m_true, m_pred, model_names, normalize=False, 
                          cmap=plt.cm.Blues, figsize=(14, 8), annotate=True, show=True):
    """A function to print and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.

    Parameters
    ----------
    m_true: np.array
        Array of true model (one-hot-encoded) indices
    m_pred: np.array
        Array of predicted model probabilities (same shape as m_true)
    model_names: list(str)
        List of model names for plotting
    normalize: bool, default: False
        Controls whether normalization shall be applied
    cmap: matplotlib.pyplot.cm.*, default: plt.cm.Blues
        Colormap
    figsize: tuple(int, int), default: (14, 8)
        Figure size
    annotate: bool, default: True
        Controls if the plot shall be annotated
    show: bool, default: True
        Controls if the plot shall be printed

    """

    # Take argmax of true and pred
    m_true = np.argmax(m_true, axis=1).astype(np.int32)
    m_pred = np.argmax(m_pred, axis=1).astype(np.int32)


    # Compute confusion matrix
    cm = confusion_matrix(m_true, m_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=model_names, yticklabels=model_names,
           ylabel='True Model',
           xlabel='Predicted Model')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.

    if annotate:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_calibration_curves(m_true, m_pred, model_names, n_bins=10, font_size=12, figsize=(12, 4)):
    """Plots the calibration curves for a model comparison problem.

    Parameters
    ----------
    cal_probs: np.array or list
        Array of calibration curve data
    model_names: list(str)
        List of model names for plotting
    font_size: int, default: 12
        Font size
    figsize: tuple(int, int), default: (12, 4)
        Figure size for plot layout

    """

    plt.rcParams['font.size'] = 12
    n_models = len(model_names)

    # Determine figure layout
    if n_models >= 6:
        n_col = int(np.sqrt(n_models))
        n_row = int(np.sqrt(n_models)) + 1
    else:
        n_col = n_models
        n_row = 1

    cal_errs, cal_probs = expected_calibration_error(m_true, m_pred, n_bins)

    # Initialize figure
    f, axarr = plt.subplots(n_row, n_col, figsize=figsize)
    if n_row > 1:
        axarr = axarr.flat

    # Loop through models
    for i, ax in enumerate(axarr.flat):

        # Plot calibration curve
        ax.plot(cal_probs[i][0], cal_probs[i][1])

        # Plot AB line
        ax.plot(ax.get_xlim(), ax.get_xlim(), '--', color='black')

        # Tweak plot
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Confidence')
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.text(0.1, 0.9, r'$\widehat{{ECE}}$ = {0:.3f}'.format(cal_errs[i]),
                        horizontalalignment='left',
                        verticalalignment='center',
                        transform=ax.transAxes,
                        size=font_size)

        # Set title
        ax.set_title(model_names[i])
    f.tight_layout()
    return f


def plot_true_est_posterior(model, n_samples, param_names, n_test=None, data_generator=None, 
                            X_test=None, theta_test=None, figsize=(15, 20), tight=True, 
                            show=True, filename=None, font_size=12):
    """
    Plots approximate posteriors.
    """

    # Plot settings
    plt.rcParams['font.size'] = font_size
    
    n_test = int(X_test.shape[0])

    # Convert theta to numpy
    #theta_test = theta_test

    # Initialize f
    f, axarr = plt.subplots(n_test, len(param_names), figsize=figsize)

    theta_samples = model.sample(X_test, n_samples)
    theta_samples_means = theta_samples.mean(axis=1)
    
    # For each row 
    for i in range(n_test):
        
        for j in range(len(param_names)):
                        
            # Plot approximate posterior
            sns.histplot(theta_samples[i, :, j], kde=True, ax=axarr[i, j], bins=15,
                            label='Estimated posterior', color='#5c92e8')
            
            # Plot lines for approximate mean, analytic mean and true data-generating value
            axarr[i, j].axvline(theta_samples_means[i, j], color='#5c92e8', label='Estimated mean')
            axarr[i, j].axvline(theta_test[i, j], color='#e55e5e', label='True')
            axarr[i, j].spines['right'].set_visible(False)
            axarr[i, j].spines['top'].set_visible(False)
            #axarr[i, j].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axarr[i, j].get_yaxis().set_ticks([])
                        
            # Set title of first row
            if i == 0:
                axarr[i, j].set_title(param_names[j])       
            
            if i == 0 and j == 0:
                f.legend(loc='lower center', bbox_to_anchor=(0.5, -0.03), shadow=True, ncol=3, fontsize=10, borderaxespad=1)
                axarr[i, j].legend(fontsize=10)
                
    if tight:
        f.tight_layout()
    f.subplots_adjust(bottom=0.12)
    # Show, if specified
    if show:
        plt.show()
    
    # Save if specified
    if filename is not None:
        f.savefig("figures/{}_{}n_density.png".format(filename, X_test.shape[1]), dpi=600, bbox_inches='tight')


def plot_posterior_comparison(param_samples, A, B, C, D, true_posterior, marginal_x, marginal_y, row, method, 
                    fontsize=13, levels=None, label_1=None, label_2=None): 
    plt.subplot(2, 3, 3*row-2) 
    if levels is not None:
        true_posterior = plt.contour(A, B, true_posterior, levels, colors='blue')
    else:
        true_posterior = plt.contour(A, B, true_posterior, colors='blue')
        #plt.clabel(true_posterior, fontsize=6, inline=1) ###
    h1, _ = true_posterior.legend_elements()
    
    # Kernel density estimator of BayesFlow samples
    a = param_samples[:, 0]
    b = param_samples[:, 1]
    ab = np.vstack([a, b])
    z = stats.gaussian_kde(ab)(ab)
    ida = z.argsort()  # Sort the points by density, so that the densest points are plotted last
    a, b, z = a[ida], b[ida], z[ida]   
    #plt.xticks([0.197, 0.198, 0.199, 0.200, 0.201])
    #plt.xticks([-0.48, -0.46, -0.44, -0.42, -0.40, -0.38, -0.36])
    #plt.yticks([-1.0, -0.8, -0.6, -0.4, -0.2])
    approximate_posterior = plt.scatter(a, b, c=z, s=30)
    h2, _ = approximate_posterior.legend_elements()
    plt.title(method, fontsize=25.5, loc='left', pad=8)
    plt.legend([h2[0], h1[0]], [label_2, label_1], fontsize=fontsize)
    plt.xlabel('Parameter $a$', fontsize=20)
    plt.ylabel('Parameter $b$', fontsize=20)

    # Check marginal densities    
    plt.subplot(2, 3, 3*row-1)
    #plt.xticks([-0.90, -0.85, -0.80, -0.75])
    #plt.yticks([0,5,10,15,20])
    plt.hist(param_samples[:, 0], bins='auto', density=1, color='orange', label=label_2) 
    plt.plot(C, marginal_x, color='b', label=label_1)
    plt.ylabel('Marginal density', fontsize=20)
    plt.xlabel('Parameter $a$', fontsize=20)
    plt.legend(fontsize=fontsize)
    
    plt.subplot(2, 3, 3*row)
    #plt.xticks([-0.46, -0.44, -0.42, -0.40, -0.38, -0.36])
    #plt.yticks([0,1,2,3,4])
    plt.hist(param_samples[:, 1], bins='auto', density=1, color='orange', label=label_2)    
    plt.plot(D, marginal_y, color='b', label=label_1)
    plt.ylabel('Marginal density', fontsize=20)
    plt.xlabel('Parameter $b$', fontsize=20)
    plt.legend(fontsize=fontsize)
    

def plot_posterior(ax, i, j, result, param_samples, method, levels=None):  
    sigma = 0.05   # noise standard deviation
    
    def prior_eval(a,b):   
        # Evaluates prior probability p(theta) according to a ~ U(0.1, 1) & shift parameter b ~ N(0, 0.25²)
        if a > 0.1 and a < 1.0:
            return 1/0.9 * norm.pdf(b, 0.0, 0.25)
        else: 
            return 0.

    def likelihood(a,b):   
        # Calculates likelihood of available data
        residual = (result[0][result[1]] - (np.sin(a*2*np.pi*result[2]) + b))/sigma
        nllh = np.sum(np.log(2*np.pi*sigma**2)+residual**2)/2
        return np.exp(-nllh)

    def unnormalized_posterior(a,b):   
        # Evaluates the unnormalized posterior probability according to Bayes' formula
        return likelihood(a,b) * prior_eval(a,b)
    
    mean_sample = np.mean(param_samples, axis=0)
    cov_sample = np.cov(param_samples.transpose())
    mean_x = mean_sample[0]
    mean_y = mean_sample[1]
    std_x = np.sqrt(cov_sample[0, 0])
    std_y = np.sqrt(cov_sample[1, 1])

    # scaling factor
    scaling_factor = dblquad(unnormalized_posterior, mean_y-10*std_y, mean_y+10*std_y, lambda y: mean_x-10*std_x, lambda y: mean_x+10*std_x)
    posterior_xy = lambda x,y: unnormalized_posterior(x,y)/scaling_factor[0]

    # Level sets of analytic posterior distribution
    grid = 201
    A = np.linspace(mean_x - 3 * std_x, mean_x + 3 * std_x, grid)
    if i==1 and j==1:
        A = np.linspace(0.948, 0.952, grid)
    B = np.linspace(mean_y - 3 * std_y, mean_y + 3 * std_y, grid)
    true_posterior = np.zeros((grid, grid))
    for iy in range(0, grid):
        for ix in range(0, grid):
            true_posterior[iy][ix] = posterior_xy(A[ix], B[iy])   
    if levels is not None:
        true_posterior = ax[i,3*j].contour(A, B, true_posterior, levels=levels, colors='blue') 
    else:
        true_posterior = ax[i,3*j].contour(A, B, true_posterior, colors='blue') 
    #plt.clabel(true_posterior, fontsize=9, inline=1)
    h1, _ = true_posterior.legend_elements()
    
    # Kernel density estimator of BayesFlow samples
    a = param_samples[:, 0]
    b = param_samples[:, 1]
    ab = np.vstack([a, b])
    z = stats.gaussian_kde(ab)(ab)
    ida = z.argsort()  # Sort the points by density, so that the densest points are plotted last
    a, b, z = a[ida], b[ida], z[ida]
    approximate_posterior = ax[i,3*j].scatter(a, b, c=z, s=25)
    h2, _ = approximate_posterior.legend_elements()
    #if j == 1:
    #    if i == 0:
    #        ax[i,3*j].legend([h2[0], h1[0]], ['BayesFlow', 'True posterior'], bbox_to_anchor=(1.04,0.61))
    #    else:
    #        ax[i,3*j].legend([h2[0], h1[0]], ['BayesFlow', 'True posterior'], bbox_to_anchor=(1.04,0.61))       
    ax[i,3*j].set_xlabel('Parameter $a$')
    ax[i,3*j].set_ylabel('Parameter $b$')  
    #if i == 0:
    #    ax[i,3*j].set_xticks([-1.25, -1.00, -0.75, -0.50, -0.25])
    #else:
    #    ax[i,3*j].set_xticks([-1.5, -1.0, -0.5, 0.0])
    #ax[i,3*j].set_yticks([-1.5, -1.0, -0.5, 0.0])
    if i == 0:
        ax[i,3*j+1].set_title(method, fontsize=14, loc='center', pad=8.5)         
        
    # Check marginal densities
    grid_x = 151
    if i==1 and j==1:
        grid_x = 1001
    grid_y = 151
    C = np.linspace(mean_x - 5 * std_x, mean_x + 5 * std_x, grid_x)
    D = np.linspace(mean_y - 5 * std_y, mean_y + 5 * std_y, grid_y )
    bounds = np.array([mean_x - 10 * std_x, mean_x + 10 * std_x, mean_y - 10 * std_y, mean_y + 10 * std_y])
    marginal_x = np.zeros(grid_x)
    for k in range(grid_x):
        x = C[k]
        integrand_y = lambda y: posterior_xy(x, y)
        marginal_x[k] = quad(integrand_y, bounds[2], bounds[3])[0]
    marginal_y = np.zeros(grid_y )
    for k in range(grid_y ):
        y = D[k]
        integrand_x = lambda x: posterior_xy(x, y)
        marginal_y[k] = quad(integrand_x, bounds[0], bounds[1])[0] 
        
    #plt.xticks([-0.90, -0.85, -0.80, -0.75])
    #plt.yticks([0,5,10,15,20])
    ax[i,3*j+1].hist(param_samples[:, 0], bins='auto', density=1, color='orange', label='BayesFlow') 
    ax[i,3*j+1].plot(C, marginal_x, color='b', label='True posterior')
    ax[i,3*j+1].set_ylabel('Marginal density')
    ax[i,3*j+1].set_xlabel('Parameter $a$')
    #ax[i,3*j+1].legend()
    
    #plt.xticks([-0.46, -0.44, -0.42, -0.40, -0.38, -0.36])
    #plt.yticks([0,1,2,3,4])
    ax[i,3*j+2].hist(param_samples[:, 1], bins='auto', density=1, color='orange', label='BayesFlow')    
    ax[i,3*j+2].plot(D, marginal_y, color='b', label='True posterior')
    ax[i,3*j+2].set_ylabel('Marginal density')
    ax[i,3*j+2].set_xlabel('Parameter $b$')
    #ax[i,3*j+2].legend()