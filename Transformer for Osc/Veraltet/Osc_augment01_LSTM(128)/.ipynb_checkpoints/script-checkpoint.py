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
summary_net = LSTM(128)

# Invertible network
inference_net = bf.networks.InvertibleNetwork(num_params=2, num_coupling_layers=4)

# Interface for density estimation and sampling
amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)

# Connect the networks with the generative model
trainer = bf.trainers.Trainer(
    amortizer=amortizer, 
    generative_model=model,
    checkpoint_path = './Osc_augment01_LSTM(128)_ckpts',
    max_to_keep=300,
    memory=False
)


# ## Training

# In[ ]:


# Online training 
losses = trainer.train_online(epochs=300, iterations_per_epoch=1000, batch_size=128, validation_sims=100)


# In[ ]:


print(losses)


# In[ ]:


fig = bf.diagnostics.plot_losses(losses['train_losses'], losses['val_losses'])


# ## Validation

# In[ ]:


fig = trainer.diagnose_latent2d()


# In[ ]:


fig = trainer.diagnose_sbc_histograms()


# In[ ]:


new_sims = trainer.configurator(model(200))


# In[ ]:


# Assumes new_sims exist from above run
posterior_draws = amortizer.sample(new_sims, n_samples=250)


# In[ ]:


fig = bf.diagnostics.plot_recovery(posterior_draws, new_sims['parameters'], param_names=['frequency', 'shift'])


# In[ ]:


fig = bf.diagnostics.plot_sbc_ecdf(posterior_draws, new_sims['parameters'])

