#!/usr/bin/env python
# coding: utf-8

# # Parameter Estimation - CR model

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from functools import partial
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
import random

from bayesflow.networks import InvertibleNetwork
from bayesflow.amortizers import SingleModelAmortizer
from bayesflow.trainers import ParameterEstimationTrainer
from bayesflow.diagnostics import *
from bayesflow.models import GenerativeModel


# ## Simulator settings

# In[3]:


def prior(batch_size):
    """
    Samples from the prior 'batch_size' times.
    ----------
    Output:
    theta : np.ndarray of shape (batch_size, theta_dim) 
    """    
    # Prior range for log-parameters: k_1, k_2 ~ N(-0.75, 0.25²) iid.
    p_samples = np.random.normal(-0.75, 0.25, size=(batch_size, 2))
    return p_samples.astype(np.float32)


# ODE model for conversion reaction  
def conversion_reaction(t, x, theta):
    theta = 10**theta
    return np.array([-theta[0]*x[0]+theta[1]*x[1], theta[0]*x[0]-theta[1]*x[1]])

x0 = [1,0]   # initial condition       
sigma = 0.015   # noise standard deviation
n_obs = 3
time_points = np.linspace(0, 10, n_obs)
missing_max = 2


def batch_simulator(prior_samples, n_obs):
    """
    Simulate multiple conversion model datasets with missing values and time labels (present time points)
    """   
    n_sim = prior_samples.shape[0]   # batch size 
    n_missing = np.random.randint(0, missing_max+1)
    n_present = n_obs - n_missing
    sim_data = np.empty((n_sim, n_present, 2), dtype=np.float32)   # 1 batch consisting of n_sim datasets, each with n_present observations
    
    for m in range(n_sim):
        theta = 10**prior_samples[m]
        s = theta[0] + theta[1]
        b = theta[0]/s
        state_2 = lambda t: b - b * np.exp(-s*t)
        
        # artificially induce missing data 
        missing_indices = random.sample(range(n_obs), n_missing)
        present_indices = np.setdiff1d(range(n_obs), missing_indices)
        present_timepoints = time_points[present_indices]
        sol = state_2(present_timepoints)
        sim_data[m, :, 0] = sol + np.random.normal(0, sigma, size = n_present)   # observable: y = x_2 + N(0,sigma²)
        sim_data[m, :, 1] = present_timepoints   # time labels
        
    return sim_data


# We build an amortized parameter estimation network.

# In[4]:


bf_meta = {
    'n_coupling_layers': 5,
    's_args': {
        'units': [64, 64, 64],
        'activation': 'elu',
        'initializer': 'glorot_uniform',
    },
    't_args': {
        'units': [64, 64, 64],
        'activation': 'elu',
        'initializer': 'glorot_uniform',
    },
    'n_params': 2
}


# In[5]:


class InvariantModule(tf.keras.Model):
    """Implements an invariant module performing a permutation-invariant transform.
    For details and rationale, see:
    [1] Bloem-Reddy, B., & Teh, Y. W. (2020). Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1. https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """

    def __init__(self, settings, **kwargs):
        """Creates an invariant module according to [1] which represents a learnable permutation-invariant
        function with an option for learnable pooling.
        Parameters
        ----------
        settings : dict
            A dictionary holding the configuration settings for the module.
        **kwargs : dict, optional, default: {}
            Optional keyword arguments passed to the `tf.keras.Model` constructor.
        """

        super().__init__(**kwargs)

        # Create internal functions
        self.s1 = Sequential([Dense(**settings["dense_s1_args"]) for _ in range(settings["num_dense_s1"])])
        self.s2 = Sequential([Dense(**settings["dense_s2_args"]) for _ in range(settings["num_dense_s2"])])

        # Pick pooling function
        if settings["pooling_fun"] == "mean":
            pooling_fun = partial(tf.reduce_mean, axis=1)
        elif settings["pooling_fun"] == "max":
            pooling_fun = partial(tf.reduce_max, axis=1)
        else:
            if callable(settings["pooling_fun"]):
                pooling_fun = settings["pooling_fun"]
            else:
                raise ConfigurationError("pooling_fun argument not understood!")
        self.pooler = pooling_fun

    def call(self, x):
        """Performs the forward pass of a learnable invariant transform.
        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, N, x_dim)
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, out_dim)
        """

        x_reduced = self.pooler(self.s1(x))
        out = self.s2(x_reduced)
        return out


class EquivariantModule(tf.keras.Model):
    """Implements an equivariant module performing an equivariant transform.
    For details and justification, see:
    [1] Bloem-Reddy, B., & Teh, Y. W. (2020). Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1. https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """

    def __init__(self, settings, **kwargs):
        """Creates an equivariant module according to [1] which combines equivariant transforms
        with nested invariant transforms, thereby enabling interactions between set members.
        Parameters
        ----------
        settings : dict
            A dictionary holding the configuration settings for the module.
        **kwargs : dict, optional, default: {}
            Optional keyword arguments passed to the ``tf.keras.Model`` constructor.
        """

        super().__init__(**kwargs)

        self.invariant_module = InvariantModule(settings)
        self.s3 = Sequential([Dense(**settings["dense_s3_args"]) for _ in range(settings["num_dense_s3"])])

    def call(self, x):
        """Performs the forward pass of a learnable equivariant transform.
        Parameters
        ----------
        x   : tf.Tensor
            Input of shape (batch_size, N, x_dim)
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, N, equiv_dim)
        """

        # Store shape of x, will be (batch_size, N, some_dim)
        shape = tf.shape(x)

        # Output dim is (batch_size, inv_dim) - > (batch_size, N, inv_dim)
        out_inv = self.invariant_module(x)

        out_inv = tf.expand_dims(out_inv, 1)
        out_inv_rep = tf.tile(out_inv, [1, shape[1], 1])

        # Concatenate each x with the repeated invariant embedding
        out_c = tf.concat([x, out_inv_rep], axis=-1)

        # Pass through equivariant func
        out = self.s3(out_c)
        return out


# In[6]:


DEFAULT_SETTING_DENSE_INVARIANT = {"units": 64, "activation": "relu", "kernel_initializer": "glorot_uniform"}

class DeepSet(tf.keras.Model):
    """Implements a deep permutation-invariant network according to [1] and [2].
    [1] Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R. R., & Smola, A. J. (2017).
    Deep sets. Advances in neural information processing systems, 30.
    [2] Bloem-Reddy, B., & Teh, Y. W. (2020).
    Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1.
    """

    def __init__(
        self,
        summary_dim=8,   # Default: 10
        num_dense_s1=2,
        num_dense_s2=2,
        num_dense_s3=2,
        num_equiv=2,
        dense_s1_args=None,
        dense_s2_args=None,
        dense_s3_args=None,
        pooling_fun="mean",
        **kwargs,
    ):
        """Creates a stack of 'num_equiv' equivariant layers followed by a final invariant layer.
        Parameters
        ----------
        summary_dim   : int, optional, default: 10
            The number of learned summary statistics.
        num_dense_s1  : int, optional, default: 2
            The number of dense layers in the inner function of a deep set.
        num_dense_s2  : int, optional, default: 2
            The number of dense layers in the outer function of a deep set.
        num_dense_s3  : int, optional, default: 2
            The number of dense layers in an equivariant layer.
        num_equiv     : int, optional, default: 2
            The number of equivariant layers in the network.
        dense_s1_args : dict or None, optional, default: None
            The arguments for the dense layers of s1 (inner, pre-pooling function). If `None`,
            defaults will be used (see `default_settings`). Otherwise, all arguments for a
            tf.keras.layers.Dense layer are supported.
        dense_s2_args : dict or None, optional, default: None
            The arguments for the dense layers of s2 (outer, post-pooling function). If `None`,
            defaults will be used (see `default_settings`). Otherwise, all arguments for a
            tf.keras.layers.Dense layer are supported.
        dense_s3_args : dict or None, optional, default: None
            The arguments for the dense layers of s3 (equivariant function). If `None`,
            defaults will be used (see `default_settings`). Otherwise, all arguments for a
            tf.keras.layers.Dense layer are supported.
        pooling_fun   : str of callable, optional, default: 'mean'
            If string argument provided, should be one in ['mean', 'max']. In addition, ac actual
            neural network can be passed for learnable pooling.
        **kwargs      : dict, optional, default: {}
            Optional keyword arguments passed to the __init__() method of tf.keras.Model.
        """

        super().__init__(**kwargs)

        # Prepare settings dictionary
        settings = dict(
            num_dense_s1=num_dense_s1,
            num_dense_s2=num_dense_s2,
            num_dense_s3=num_dense_s3,
            dense_s1_args=DEFAULT_SETTING_DENSE_INVARIANT if dense_s1_args is None else dense_s1_args,
            dense_s2_args=DEFAULT_SETTING_DENSE_INVARIANT if dense_s2_args is None else dense_s2_args,
            dense_s3_args=DEFAULT_SETTING_DENSE_INVARIANT if dense_s3_args is None else dense_s3_args,
            pooling_fun=pooling_fun,
        )

        # Create equivariant layers and final invariant layer
        self.equiv_layers = Sequential([EquivariantModule(settings) for _ in range(num_equiv)])
        self.inv = InvariantModule(settings)

        # Output layer to output "summary_dim" learned summary statistics
        self.out_layer = Dense(summary_dim, activation="linear")
        self.summary_dim = summary_dim

    def call(self, x):
        """Performs the forward pass of a learnable deep invariant transformation consisting of
        a sequence of equivariant transforms followed by an invariant transform.
        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, n_obs, data_dim)
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, out_dim)
        """

        # Pass through series of augmented equivariant transforms
        out_equiv = self.equiv_layers(x)

        # Pass through final invariant layer
        out = self.out_layer(self.inv(out_equiv))

        return out


# In[7]:


summary_net = DeepSet()
inference_net = InvertibleNetwork(bf_meta)
amortizer = SingleModelAmortizer(inference_net, summary_net)


# We connect the prior and simulator through a *GenerativeModel* class which will take care of forward inference.

# In[8]:


generative_model = GenerativeModel(prior, batch_simulator)


# In[9]:


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=2000,
    decay_rate=0.95,
    staircase=True,
)


# In[10]:


trainer = ParameterEstimationTrainer(
    network=amortizer, 
    generative_model=generative_model,
    learning_rate = lr_schedule,
    checkpoint_path = './CR3_timelabels_5ACB_[64,64,64]_invariant(8)_ckpts',
    max_to_keep=300,
    skip_checks=True
)


# ### Online training

# In[ ]:


losses = trainer.train_online(epochs=300, iterations_per_epoch=1000, batch_size=128, n_obs=n_obs)
print(losses)
