#!/usr/bin/env python
# coding: utf-8

# # Oscillatory model

# In[1]:


import numpy as np
import random
import bayesflow as bf
from tensorflow.keras.layers import LSTM


# ## Simulator settings

# In[2]:


def batch_prior(batch_size):
    """
    Samples from the prior 'batch_size' times.
    ----------
    Output:
    theta : np.ndarray of shape (batch_size, theta_dim) 
    """    
    # Prior range: frequency parameter a ~ U(0.1, 1) & shift parameter b ~ N(0, 0.25²)
    freq_samples = np.random.uniform(0.1, 1.0, size=(batch_size, 1))
    shift_samples = np.random.normal(0.0, 0.25, size=(batch_size, 1))
    p_samples = np.c_[freq_samples, shift_samples]
    return p_samples.astype(np.float32)


def batch_simulator(prior_samples, n_obs=41, t_end=10, sigma=0.05, missing_max=21, **kwargs):   
    """Simulate multiple oscillatory model data sets with missing values and binary indicator augmentation""" 

    n_sim = prior_samples.shape[0]   # batch size    
    n_missing = np.random.randint(0, missing_max+1, size=n_sim)
    sim_data = np.ones((n_sim, n_obs, 2), dtype=np.float32)   # 1 batch consisting of n_sim data sets, each with n_obs observations
    time_points = np.linspace(0, t_end, n_obs)
    
    for m in range(n_sim):        
        a = prior_samples[m, 0]   # frequency
        b = prior_samples[m, 1]   # shift
        sim_data[m, :, 0] = np.sin(a*2*np.pi*time_points) + b + np.random.normal(0, sigma, size=n_obs)
        
        # artificially induce missing data
        missing_indices = random.sample(range(n_obs), n_missing[m])
        sim_data[m][missing_indices] = np.array([-5.0, 0.0])
        
    return sim_data   


# ## Generative Model

# In[3]:


model = bf.simulation.GenerativeModel(batch_prior, batch_simulator, prior_is_batched=True, simulator_is_batched=True)


# ## Network setup

# In[4]:


# Summary network
summary_net = bf.networks.TimeSeriesTransformer(
        input_dim=2,
        attention_settings=dict(num_heads=2, key_dim=16),
        dense_settings=dict(units=64, activation='relu'),
)

# Invertible network
inference_net = bf.networks.InvertibleNetwork(num_params=2, num_coupling_layers=4)

# Interface for density estimation and sampling
amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)

# Connect the networks with the generative model
trainer = bf.trainers.Trainer(
    amortizer=amortizer, 
    generative_model=model,
    checkpoint_path = './Osc_augment01_trafo_ckpts',
    max_to_keep=300
)


# ## Training

# In[ ]:


# Online training 
losses = trainer.train_online(epochs=300, iterations_per_epoch=1000, batch_size=128, validation_sims=100)
