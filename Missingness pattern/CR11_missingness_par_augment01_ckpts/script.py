#!/usr/bin/env python
# coding: utf-8

# # CR model with missingness parameter

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM
import random

from bayesflow.networks import InvertibleNetwork 
from bayesflow.amortizers import SingleModelAmortizer
from bayesflow.trainers import ParameterEstimationTrainer
from bayesflow.models import GenerativeModel


# ## Simulator settings

# In[89]:


def prior(batch_size):
    """
    Samples from the prior 'batch_size' times.
    ----------
    Output:
    theta : np.ndarray of shape (batch_size, theta_dim) 
    """    
    # log-rate parameter k1
    k1_samples = np.random.normal(-0.75, 0.25, size=batch_size)
    # missingness parameter p
    p_samples = np.random.uniform(0, 0.8, size=batch_size)
    theta_samples = np.c_[k1_samples, p_samples]
    return theta_samples.astype(np.float32)

c2 = 10**(-0.75)   # fixed parameter k2=-0.75
sigma = 0.015   # noise standard deviation
n_obs = 11
time_points = np.linspace(0, 10, n_obs)

def batch_simulator(prior_samples, n_obs):
    """
    Simulate multiple CR model data sets with floor(p*n_obs) missing values and binary indicator augmentation
    """   
    n_sim = prior_samples.shape[0]   # batch size 
    sim_data = np.ones((n_sim, n_obs, 2), dtype=np.float32)   # 1 batch consisting of n_sim data sets, each with n_obs observations
    c1 = 10**prior_samples[:, 0]
    n_missing = np.floor(prior_samples[:, 1] * n_obs).astype(int)
    
    for m in range(n_sim):
        s = c1[m] + c2
        b = c1[m]/s
        state_2 = lambda t: b - b * np.exp(-s*t)
        sol = state_2(time_points)
        sim_data[m, :, 0] = sol + np.random.normal(0, sigma, size = n_obs)   # observable: y = x_2 + N(0,sigma²) 
        
        # artificially induce missing data
        missing_indices = random.sample(range(n_obs), n_missing[m])
        sim_data[m][missing_indices] = np.array([-1.0, 0.0])  
        
    return sim_data


# We build an amortized parameter estimation network.

# In[100]:


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


# In[101]:


summary_net = LSTM(32)
inference_net = InvertibleNetwork(bf_meta)
amortizer = SingleModelAmortizer(inference_net, summary_net)


# We connect the prior and simulator through a *GenerativeModel* class which will take care of forward inference.

# In[102]:


generative_model = GenerativeModel(prior, batch_simulator)


# In[103]:


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=2000,
    decay_rate=0.95,
    staircase=True,
)


# In[104]:


trainer = ParameterEstimationTrainer(
    network=amortizer, 
    generative_model=generative_model,
    learning_rate = lr_schedule,
    checkpoint_path = './CR11_missingness_par_augment01_ckpts',
    max_to_keep=300,
    skip_checks=True
)


# ### Online training

# In[ ]:


losses = trainer.train_online(epochs=300, iterations_per_epoch=1000, batch_size=128, n_obs=n_obs)
print(losses)
