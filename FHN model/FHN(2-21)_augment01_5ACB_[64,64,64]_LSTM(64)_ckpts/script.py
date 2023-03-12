#!/usr/bin/env python
# coding: utf-8

# # Parameter Estimation - FitzHugh-Nagumo model

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from scipy import integrate
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
    # Prior range for log-parameters:
    p_samples = np.random.uniform(low=-2, high=0, size=(batch_size, 3))
    return p_samples.astype(np.float32)

# ODE model 
def fhn_dynamics(t, x, theta):
    theta = 10**theta
    return np.array([theta[2]*(x[0]-1/3*x[0]**3+x[1]), -1/theta[2]*(x[0]-theta[0]+theta[1]*x[1])])

x0 = [-1, 1]
t_end = 15
n_obs = 21
n_min = 2
time_points = np.linspace(0, t_end, n_obs)
sigma = 0.05   # noise standard deviation

def batch_simulator(prior_samples, n_obs):
    """
    Simulate a batch of FHN data sets with missing values at the end and binary augmentation
    """
    n_sim = prior_samples.shape[0]  # batch size
    sim_data = np.ones((n_sim, n_obs, 2), dtype=np.float32)  # 1 batch consisting of n_sim data sets, each with n_obs observations
    n_present = np.random.randint(n_min, n_obs + 1, size=n_sim)

    for m in range(n_sim):
        rhs = lambda x,t: fhn_dynamics(t, x, prior_samples[m])
        sol = integrate.odeint(rhs, x0, time_points[0:n_present[m]])
        sim_data[m, 0:n_present[m], 0] = sol[:,0] + np.random.normal(0, sigma, size=n_present[m])

        # artificially induce missing data
        sim_data[m][n_present[m]:n_obs] = np.array([-5.0, 0.0])
    
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
    'n_params': 3
}


# In[5]:


summary_net = LSTM(64)
inference_net = InvertibleNetwork(bf_meta)
amortizer = SingleModelAmortizer(inference_net, summary_net)


# We connect the prior and simulator through a *GenerativeModel* class which will take care of forward inference.

# In[6]:


generative_model = GenerativeModel(prior, batch_simulator)


# In[7]:


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,   
    decay_steps=2000,
    decay_rate=0.95,
    staircase=True,
)


# In[8]:


trainer = ParameterEstimationTrainer(
    network=amortizer, 
    generative_model=generative_model,
    learning_rate = lr_schedule,
    checkpoint_path = './FHN(2-21)_augment01_5ACB_[64,64,64]_LSTM(64)_ckpts',
    max_to_keep=300,
    skip_checks=True
)


# ### Online training

# In[ ]:


losses = trainer.train_online(epochs=300, iterations_per_epoch=1000, batch_size=64, n_obs=n_obs)
print(losses)
