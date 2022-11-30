import numpy as np
import tensorflow as tf
from functools import partial
from sklearn.metrics import r2_score


def gaussian_kernel_matrix(x, y, sigmas):
    """
    Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
    x: a tensor of shape [num_samples, num_features]
    y: a tensor of shape [num_samples, num_features]
    sigmas: a tensor of floats which denote the widths of each of the
      gaussians in the kernel.
    Returns:
    A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """

    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    dist = tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))
    

def mmd_kernel(x, y, kernel=gaussian_kernel_matrix):
    """
    Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y.
    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.
    Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
    """

    loss = tf.reduce_mean(kernel(x, x))
    loss += tf.reduce_mean(kernel(y, y))
    loss -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    loss = tf.where(loss > 0, loss, 0)
    return loss


def maximum_mean_discrepancy(source_samples, target_samples, weight=1., minimum=0.):
    """
    This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
    different Gaussian kernels.
    ----------
    Arguments:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the MMD loss.
    Returns:
    a scalar tensor representing the MMD loss value.
    """

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=sigmas)
    loss_value = mmd_kernel(source_samples, target_samples, kernel=gaussian_kernel)
    loss_value = tf.maximum(minimum, loss_value) * weight
    return loss_value


def calibration_error(theta_samples, theta_test, alpha_resolution=100):
    """
    Computes the calibration error of an approximate posterior per parameters.
    The calibration error is given as the median of the absolute deviation
    between alpha (0 - 1) (credibility level) and the relative number of inliers from
    theta test.
    
    ----------
    
    Arguments:
    theta_samples       : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test          : np.ndarray of shape (n_test, n_params) -- the 'true' test values
    alpha_resolution    : int -- the number of intervals to consider 
    
    ----------
    
    Returns:
    
    cal_errs  : np.ndarray of shape (n_params, ) -- the calibration errors per parameter
    """

    n_params = theta_test.shape[1]
    n_test = theta_test.shape[0]
    alphas = np.linspace(0.01, 1.0, alpha_resolution)
    cal_errs = np.zeros(n_params)
    
    # Loop for each parameter
    for k in range(n_params):
        alphas_in = np.zeros(len(alphas))
        # Loop for each alpha
        for i, alpha in enumerate(alphas):

            # Find lower and upper bounds of posterior distribution
            region = 1 - alpha
            lower = np.round(region / 2, 3)
            upper = np.round(1 - (region / 2), 3)

            # Compute quantiles for given alpha using the entire sample
            quantiles = np.quantile(theta_samples[:, :, k], [lower, upper], axis=0).T

            # Compute the relative number of inliers
            inlier_id = (theta_test[:, k] > quantiles[:, 0]) &  (theta_test[:, k] < quantiles[:, 1])
            inliers_alpha = np.sum(inlier_id) / n_test
            alphas_in[i] = inliers_alpha
        
        # Compute calibration error for k-th parameter
        diff_alphas = np.abs(alphas - alphas_in)
        cal_err = np.round(np.median(diff_alphas), 3)
        cal_errs[k] = cal_err
        
    return cal_errs


def rmse(theta_samples, theta_test, normalized=True):
    """
    Computes the RMSE or normalized RMSE (NRMSE) between posterior means 
    and true parameter values for each parameter
    
    ----------
    
    Arguments:
    theta_samples   : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test      : np.ndarray of shape (n_test, n_params) -- the 'true' test values
    normalized      : boolean -- whether to compute nrmse or rmse (default True)
    
    ----------
    
    Returns:
    
    (n)rmse  : np.ndarray of shape (n_params, ) -- the (n)rmse per parameter
    """

    # Convert tf.Tensors to numpy, if passed
    if type(theta_samples) is not np.ndarray:
        theta_samples = theta_samples.numpy() 
    if type(theta_test) is not np.ndarray:
        theta_test = theta_test.numpy()
    
    theta_approx_means = theta_samples.mean(0)
    rmse = np.sqrt( np.mean( (theta_approx_means - theta_test)**2, axis=0) )
    
    if normalized:
        rmse = rmse / (theta_test.max(axis=0) - theta_test.min(axis=0))
    return rmse


def R2(theta_samples, theta_test):
    
    """
    Computes the R^2 score as a measure of reconstruction (percentage of variance
    in true parameters captured by estimated parameters)
    
    ----------
    Arguments:
    theta_samples   : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test      : np.ndarray of shape (n_test, n_params) -- the 'true' test values
    
    ----------
    Returns:
    
    r2s  : np.ndarray of shape (n_params, ) -- the r2s per parameter
    """

    # Convert tf.Tensors to numpy, if passed
    if type(theta_samples) is not np.ndarray:
        theta_samples = theta_samples.numpy() 
    if type(theta_test) is not np.ndarray:
        theta_test = theta_test.numpy()
    
    theta_approx_means = theta_samples.mean(0)
    return r2_score(theta_test, theta_approx_means, multioutput='raw_values')


def resimulation_error_original(theta_samples, theta_test, simulator, **sim_args):
    """
    Computes the median deviation between data simulated with true true test parameters
    and data simulated with estimated parameters.
    
    ----------
    Arguments:
    theta_samples   : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test      : np.ndarray of shape (n_test, n_params) -- the 'true' test values
    simulator       : callable -- the simulator object taking a matrix or (1, n_params) vector
                                  of parameters and returning a 3D tensor of shape (n_test, n_points, dim)
    sim_args        : arguments for the simulator
    
    ----------
    
    Returns:
    (n)rmse  : np.ndarray of shape (n_params, ) -- the (n)rmse per parameter
    """

    # Convert tf.Tensors to numpy, if passed
    if type(theta_samples) is not np.ndarray:
        theta_samples = theta_samples.numpy() 
    if type(theta_test) is not np.ndarray:
        theta_test = theta_test.numpy()
    
    theta_approx_means = theta_samples.mean(0)
    n_test = theta_test.shape[0]

    # Simulate with true and estimated
    X_test_true = simulator(theta_test, **sim_args)
    X_test_est = simulator(theta_approx_means, **sim_args)

    # Compute MMDs
    mmds = [maximum_mean_discrepancy(X_test_true[i], X_test_est[i]) for i in range(n_test)]
    return np.median(mmds)


def resimulation_error(theta_samples, theta_test, simulator, X_test_true, **sim_args):
    """
    Computes the median deviation between data simulated with true true test parameters
    and data simulated with estimated parameters.
    
    ----------
    Arguments:
    theta_samples   : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test      : np.ndarray of shape (n_test, n_params) -- the 'true' test values
    simulator       : callable -- the simulator object taking a matrix or (1, n_params) vector
                                  of parameters and returning a 3D tensor of shape (n_test, n_points, dim)
    X_test_true     : np.ndarray of shape (n_test, n_points, dim) -- the 'true' datasets
    sim_args        : arguments for the simulator
    
    ----------
    
    Returns:
    (n)rmse  : np.ndarray of shape (n_params, ) -- the (n)rmse per parameter
    """

    # Convert tf.Tensors to numpy, if passed
    if type(theta_samples) is not np.ndarray:
        theta_samples = theta_samples.numpy() 
    if type(theta_test) is not np.ndarray:
        theta_test = theta_test.numpy()
    
    theta_approx_means = theta_samples.mean(0)
    n_test = theta_test.shape[0]

    # Simulate with true and estimated
    X_test_est = simulator(theta_approx_means, **sim_args)

    # Compute MMDs
    mmds = [maximum_mean_discrepancy(X_test_true[i], X_test_est[i]) for i in range(n_test)]
    return np.median(mmds)


def bootstrap_metrics_original(theta_samples, theta_test, simulator, X_test_true, p_bar=None, n_bootstrap=100, **simulator_args):
    """
    Computes bootstrap diagnostic metrics for samples from the approximate posterior.
    
    ----------
    Arguments:
    theta_samples   : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test      : np.ndarray of shape (n_test, n_params) -- the 'true' test values
    simulator       : callable -- the simulator object taking a matrix or (1, n_params) vector
                                  of parameters and returning a 3D tensor of shape (n_test, n_points, dim)
    X_test_true     : np.ndarray of shape (n_test, n_points, dim) -- the 'true' datasets
    p_bar           : progressbar or None
    n_bootstrap     : int -- the number of bootstrap samples to take 
    simulator_args  : arguments for the simulator
    
    ----------
    
    Returns:
    (n)rmse  : np.ndarray of shape (n_params, ) -- the (n)rmse per parameter
    """
    
    n_params = int(theta_test.shape[1])
    n_test = int(theta_test.shape[0])
    
    metrics = {
        'cal_err': [],
        'rmse': [],
        'r2': [],
        'res_err': []
    }
    
    for bi in range(n_bootstrap):
        
        # Get bootstrap samples
        b_idx = np.random.choice(np.random.permutation(n_test), size=n_test, replace=True)
        theta_test_b = tf.gather(theta_test, b_idx, axis=0).numpy()
        theta_samples_b = tf.gather(theta_samples, b_idx, axis=1).numpy()
        
        # Obtain metrics on bootstrap sample
        cal_errs = calibration_error(theta_samples_b, theta_test_b)
        nrmses = rmse(theta_samples_b, theta_test_b)
        r2s = R2(theta_samples_b, theta_test_b)
        res_err = resimulation_error(theta_samples_b, theta_test_b, simulator, X_test_true, **simulator_args)
        
        # Add to dict
        metrics['cal_err'].append(cal_errs)
        metrics['rmse'].append(nrmses)
        metrics['r2'].append(r2s)
        metrics['res_err'].append(res_err)
        
        if p_bar is not None:
            p_bar.set_postfix_str("Bootstrap sample {}".format(bi+1))
            p_bar.update(1)
      
    # Convert to arrays for convenience
    metrics = {k: np.array(v) for k, v in metrics.items()}
    return metrics


def bootstrap_metrics(theta_samples, theta_test, p_bar=None, n_bootstrap=100):
    """
    Computes bootstrap diagnostic metrics for samples from the approximate posterior.
    
    ----------
    Arguments:
    theta_samples   : np.ndarray of shape (n_samples, n_test, n_params) -- the samples from
                          the approximate posterior
    theta_test      : np.ndarray of shape (n_test, n_params) -- the 'true' test values
    p_bar           : progressbar or None
    n_bootstrap     : int -- the number of bootstrap samples to take 
    
    ----------
    
    Returns:
    (n)rmse  : np.ndarray of shape (n_params, ) -- the (n)rmse per parameter
    """
    
    n_params = int(theta_test.shape[1])
    n_test = int(theta_test.shape[0])
    
    metrics = {
        #'cal_err': [],
        'rmse': [],
        'r2': []#,
        #'res_err': []
    }
    
    for bi in range(n_bootstrap):
        
        # Get bootstrap samples
        b_idx = np.random.choice(np.random.permutation(n_test), size=n_test, replace=True)
        theta_test_b = tf.gather(theta_test, b_idx, axis=0).numpy()
        theta_samples_b = tf.gather(theta_samples, b_idx, axis=1).numpy()
        
        # Obtain metrics on bootstrap sample
        #cal_errs = calibration_error(theta_samples_b, theta_test_b)
        nrmses = rmse(theta_samples_b, theta_test_b)
        r2s = R2(theta_samples_b, theta_test_b)
        #res_err = resimulation_error(theta_samples_b, theta_test_b, simulator, X_test_true, **simulator_args)
        
        # Add to dict
        #metrics['cal_err'].append(cal_errs)
        metrics['rmse'].append(nrmses)
        metrics['r2'].append(r2s)
        #metrics['res_err'].append(res_err)
        
        if p_bar is not None:
            p_bar.set_postfix_str("Bootstrap sample {}".format(bi+1))
            p_bar.update(1)
      
    # Convert to arrays for convenience
    metrics = {k: np.array(v) for k, v in metrics.items()}
    return metrics


def display_metrics(metrics):
    #cal_err = metrics['cal_err']
    #cal_err_1_mean = np.mean(cal_err[:,0])
    #cal_err_1_std = np.std(cal_err[:,0])
    #cal_err_2_mean = np.mean(cal_err[:,1])
    #cal_err_2_std = np.std(cal_err[:,1])
    
    nrmse = metrics['rmse']
    nrmse_1_mean = np.mean(nrmse[:,0])
    nrmse_1_std = np.std(nrmse[:,0])
    nrmse_2_mean = np.mean(nrmse[:,1])
    nrmse_2_std = np.std(nrmse[:,1])    
    
    r2 = metrics['r2']
    r2_1_mean = np.mean(r2[:,0])
    r2_1_std = np.std(r2[:,0])
    r2_2_mean = np.mean(r2[:,1])
    r2_2_std = np.std(r2[:,1])    
    
    #res_err = metrics['res_err']
    #res_err_mean = np.mean(res_err)
    #res_err_std = np.std(res_err)
   
    #print('Err_cal(k_1): {:.3f} \u00B1 {:.3f}'.format(cal_err_1_mean, cal_err_1_std))
    #print('Err_cal(k_2): {:.3f} \u00B1 {:.3f}'.format(cal_err_2_mean, cal_err_2_std))
    print('NRMSE(k_1): {:.3f} \u00B1 {:.3f}'.format(nrmse_1_mean, nrmse_1_std))
    print('NRMSE(k_2): {:.3f} \u00B1 {:.3f}'.format(nrmse_2_mean, nrmse_2_std))
    print('R²(k_1): {:.3f} \u00B1 {:.3f}'.format(r2_1_mean, r2_1_std))
    print('R²(k_2): {:.3f} \u00B1 {:.3f}'.format(r2_2_mean, r2_2_std))
    #print('Err_sim: {:.2f} \u00B1 {:.2f}'.format(res_err_mean, res_err_std))
    
    return np.array([[nrmse_1_mean, nrmse_2_mean, r2_1_mean, r2_2_mean], [nrmse_1_std, nrmse_2_std, r2_1_std, r2_2_std]])
    
    
def bar_chart(ax, means, column, category, se=None, y_lim=None, y_ticks=None):
    ax[column].set_title(category, fontsize=24, pad=8)
    if se is not None:
        ax[column].bar(x=[1,2,3], height=means, yerr=se, capsize=10, width=1.0, color=['orange', 'aquamarine', 'dodgerblue'])
    else: 
        ax[column].bar(x=[1], height=means[0], width=1.0, color=['orange'], label='Augment by $0/1$')
        ax[column].bar(x=[2], height=means[1], width=1.0, color=['aquamarine'], label='Insert $-1$')
        ax[column].bar(x=[3], height=means[2], width=1.0, color=['dodgerblue'], label='Time labels')
    ax[column].set_xticks([1,2,3])
    ax[column].set_xticklabels(['','',''])
    ax[column].tick_params(bottom=False)
    if y_lim is not None:
        ax[column].set_ylim(y_lim)
    if y_ticks is not None:
        ax[column].set_yticks(y_ticks)