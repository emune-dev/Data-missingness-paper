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
from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Sequential
from warnings import warn
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


# In[8]:


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
        return 


# In[9]:


class MultiHeadAttentionBlock(tf.keras.Model):
    """Implements the MAB block from [1] which represents learnable cross-attention.
    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019). 
        Set transformer: A framework for attention-based permutation-invariant neural networks. 
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(
        self,
        input_dim,
        attention_settings, 
        num_dense_fc,
        dense_settings, 
        use_layer_norm,
        **kwargs
    ):
        """Creates a multihead attention block which will typically be used as part of a 
        set transformer architecture according to [1].
        Parameters
        ----------
        input_dim           : int
            The dimensionality of the input data (last axis).
        attention_settings  : dict
            A dictionary which will be unpacked as the arguments for the ``MultiHeadAttention`` layer
            See https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention.
        num_dense_fc        : int
            The number of hidden layers for the internal feedforward network
        dense_settings      : dict
            A dictionary which will be unpacked as the arguments for the ``Dense`` layer
        use_layer_norm      : boolean
            Whether layer normalization before and after attention + feedforward
        **kwargs            : dict, optional, default: {}
            Optional keyword arguments passed to the __init__() method of tf.keras.Model
        """

        super().__init__(**kwargs)

        self.att = MultiHeadAttention(**attention_settings)
        self.ln_pre = LayerNormalization() if use_layer_norm else None
        self.fc = Sequential([Dense(**dense_settings) for _ in range(num_dense_fc)])
        self.fc.add(Dense(input_dim))
        self.ln_post = LayerNormalization() if use_layer_norm else None

    def call(self, x, y, **kwargs):
        """Performs the forward pass through the attention layer.
        Parameters
        ----------
        x : tf.Tensor
            Input of shape (batch_size, set_size_x, input_dim)
        y : tf.Tensor
            Input of shape (batch_size, set_size_y, input_dim)
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, set_size_x, input_dim)
        """

        h = x + self.att(x, y, y, **kwargs)
        if self.ln_pre is not None:
            h = self.ln_pre(h, **kwargs)
        out = h + self.fc(h, **kwargs)
        if self.ln_post is not None:
            out = self.ln_post(out, **kwargs)
        return out


class SelfAttentionBlock(tf.keras.Model):
    """Implements the SAB block from [1] which represents learnable self-attention.
    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019). 
        Set transformer: A framework for attention-based permutation-invariant neural networks. 
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(
        self, 
        input_dim, 
        attention_settings, 
        num_dense_fc, 
        dense_settings, 
        use_layer_norm, 
        **kwargs
    ):
        """Creates a self-attention attention block which will typically be used as part of a 
        set transformer architecture according to [1].
        Parameters
        ----------
        input_dim           : int
            The dimensionality of the input data (last axis).
        attention_settings  : dict
            A dictionary which will be unpacked as the arguments for the ``MultiHeadAttention`` layer
            See https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention.
        num_dense_fc        : int
            The number of hidden layers for the internal feedforward network
        dense_settings      : dict
            A dictionary which will be unpacked as the arguments for the ``Dense`` layer
        use_layer_norm      : boolean
            Whether layer normalization before and after attention + feedforward
        **kwargs            : dict, optional, default: {}
            Optional keyword arguments passed to the __init__() method of tf.keras.Model
        """


        super().__init__(**kwargs)

        self.mab = MultiHeadAttentionBlock(
            input_dim, attention_settings, num_dense_fc, dense_settings, use_layer_norm)

    def call(self, x, **kwargs):
        """Performs the forward pass through the self-attention layer.
        Parameters
        ----------
        x   : tf.Tensor
            Input of shape (batch_size, set_size, input_dim)
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, set_size, input_dim)
        """

        return self.mab(x, x, **kwargs)


class InducedSelfAttentionBlock(tf.keras.Model):
    """Implements the ISAB block from [1] which represents learnable self-attention specifically
    designed to deal with large sets via a learnable set of "inducing points".
    
    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019). 
        Set transformer: A framework for attention-based permutation-invariant neural networks. 
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(
        self, 
        input_dim, 
        attention_settings, 
        num_dense_fc,
        dense_settings, 
        use_layer_norm, 
        num_inducing_points, 
        **kwargs
    ):
        """Creates a self-attention attention block with inducing points (ISAB) which will typically 
        be used as part of a set transformer architecture according to [1].
        Parameters
        ----------
        input_dim           : int
            The dimensionality of the input data (last axis).
        attention_settings  : dict
            A dictionary which will be unpacked as the arguments for the ``MultiHeadAttention`` layer
            See https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention.
        num_dense_fc        : int
            The number of hidden layers for the internal feedforward network
        dense_settings      : dict
            A dictionary which will be unpacked as the arguments for the ``Dense`` layer
        use_layer_norm      : boolean
            Whether layer normalization before and after attention + feedforward
        num_inducing_points : int
            The number of inducing points. Should be lower than the smallest set size
        **kwargs            : dict, optional, default: {}
            Optional keyword arguments passed to the __init__() method of tf.keras.Model
        """

        super().__init__(**kwargs)
        
        init = tf.keras.initializers.GlorotUniform()
        self.I = tf.Variable(init(shape=(num_inducing_points, input_dim)), name='I', trainable=True)
        self.mab0 = MultiHeadAttentionBlock(
            input_dim, attention_settings, num_dense_fc, dense_settings, use_layer_norm)
        self.mab1 = MultiHeadAttentionBlock(
            input_dim, attention_settings, num_dense_fc, dense_settings, use_layer_norm)

    def call(self, x, **kwargs):
        """Performs the forward pass through the self-attention layer.
        Parameters
        ----------
        x   : tf.Tensor
            Input of shape (batch_size, set_size, input_dim)
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, set_size, input_dim)
        """

        batch_size = x.shape[0]
        h = self.mab0(tf.stack([self.I] * batch_size), x, **kwargs)
        return self.mab1(x, h, **kwargs)


class PoolingWithAttention(tf.keras.Model):
    """Implements the pooling with multihead attention (PMA) block from [1] which represents 
    a permutation-invariant encoder for set-based inputs.
    
    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019). 
        Set transformer: A framework for attention-based permutation-invariant neural networks. 
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(
        self,
        summary_dim, 
        attention_settings,
        num_dense_fc,
        dense_settings,
        use_layer_norm, 
        num_seeds=1,
        **kwargs
    ):
        """Creates a multihead attention block (MAB) which will perform cross-attention between an input set
        and a set of seed vectors (typically one for a single summary) with summary_dim output dimensions.
        Could also be used as part of a ``DeepSet`` for representing learnabl instead of fixed pooling.
        Parameters
        ----------
        summary_dim         : int
            The dimensionality of the learned permutation-invariant representation.
        attention_settings  : dict
            A dictionary which will be unpacked as the arguments for the ``MultiHeadAttention`` layer
            See https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention.
        num_dense_fc        : int
            The number of hidden layers for the internal feedforward network
        dense_settings      : dict
            A dictionary which will be unpacked as the arguments for the ``Dense`` layer
        use_layer_norm      : boolean
            Whether layer normalization before and after attention + feedforward
        num_seeds           : int, optional, default: 1
            The number of "seed vectors" to use. Each seed vector represents a permutation-invariant
            summary of the entire set. If you use ``num_seeds > 1``, the resulting seeds will be flattened
            into a 2-dimensional output, which will have a dimensionality of ``num_seeds * summary_dim``
        **kwargs            : dict, optional, default: {}
            Optional keyword arguments passed to the __init__() method of tf.keras.Model
        """

        super().__init__(**kwargs)

        self.mab = MultiHeadAttentionBlock(
            summary_dim, attention_settings, num_dense_fc, dense_settings, use_layer_norm,  **kwargs
        )
        init = tf.keras.initializers.GlorotUniform()
        self.seed_vec = init(shape=(num_seeds, summary_dim))
        self.fc = Sequential([Dense(**dense_settings) for _ in range(num_dense_fc)])
        self.fc.add(Dense(summary_dim))

    def call(self, x, **kwargs):
        """Performs the forward pass through the PMA block.
        Parameters
        ----------
        x   : tf.Tensor
            Input of shape (batch_size, set_size, input_dim)
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, num_seeds * summary_dim)
        """

        batch_size = x.shape[0]
        out = self.fc(x)
        out = self.mab(tf.stack([self.seed_vec] * batch_size), out, **kwargs)
        return tf.reshape(out, (out.shape[0], -1))


# In[10]:


class SetTransformer(tf.keras.Model):
    """Implements the set transformer architecture from [1] which ultimately represents
    a learnable permutation-invariant function.
    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019). 
        Set transformer: A framework for attention-based permutation-invariant neural networks. 
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(
        self,
        input_dim,
        attention_settings,
        dense_settings,
        use_layer_norm=True, 
        num_dense_fc=2,
        summary_dim=8,   # Default: 10
        num_attention_blocks=2, 
        num_inducing_points=32,
        num_seeds=1,
        **kwargs
    ):
        """Creates a set transformer architecture according to [1] which will extract permutation-invariant
        features from an input set using a set of seed vectors (typically one for a single summary) with ``summary_dim`` 
        output dimensions.
        Parameters
        ----------
        input_dim            : int
            The dimensionality of the input data (last axis).
        attention_settings   : dict
            A dictionary which will be unpacked as the arguments for the ``MultiHeadAttention`` layer
            For instance, to use an attention block with 4 heads and key dimension 32, you can do:
    
            ``attention_settings=dict(num_heads=4, key_dim=32)``
            You may also want to include dropout regularization in small-to-medium data regimes:
            ``attention_settings=dict(num_heads=4, key_dim=32, dropout=0.1)``
            For more details and arguments, see:
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
        dense_settings       : dict
            A dictionary which will be unpacked as the arguments for the ``Dense`` layer.
            For instance, to use hidden layers with 32 units and a relu activation, you can do:
            ``dict(units=32, activation='relu')
            For more details and arguments, see:
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        use_layer_norm       : boolean, optional, default: True 
            Whether layer normalization before and after attention + feedforward
        num_dense_fc         : int, optional, default: 2
            The number of hidden layers for the internal feedforward network
        summary_dim          : int
            The dimensionality of the learned permutation-invariant representation.
        num_attention_blocks : int, optional, default: 2
            The number of self-attention blocks to use before pooling.
        num_inducing_points  : int or None, optional, default: 32
            The number of inducing points. Should be lower than the smallest set size. 
            If ``None`` selected, a vanilla self-attenion block (SAB) will be used, otherwise
            ISAB blocks will be used. For ``num_attention_blocks > 1``, we currently recommend
            always using some number of inducing points.
        num_seeds            : int, optional, default: 1
            The number of "seed vectors" to use. Each seed vector represents a permutation-invariant
            summary of the entire set. If you use ``num_seeds > 1``, the resulting seeds will be flattened
            into a 2-dimensional output, which will have a dimensionality of ``num_seeds * summary_dim``.
        **kwargs             : dict, optional, default: {}
            Optional keyword arguments passed to the __init__() method of tf.keras.Model
        """

        super().__init__(**kwargs)

        # Construct a series of self-attention blocks
        self.attention_blocks = Sequential()
        for _ in range(num_attention_blocks):
            if num_inducing_points is not None:
                block = InducedSelfAttentionBlock(input_dim, attention_settings, num_dense_fc,
                    dense_settings, use_layer_norm, num_inducing_points)
            else:
                block = SelfAttentionBlock(input_dim, attention_settings, num_dense_fc,
                    dense_settings, use_layer_norm)
                self.attention_blocks.add(block)

        # Pooler will be applied to the representations learned through self-attention
        self.pooler = PoolingWithAttention(summary_dim, attention_settings, num_dense_fc,
                dense_settings, use_layer_norm, num_seeds)

    def call(self, x, **kwargs):
        """Performs the forward pass through the set-transformer.
        Parameters
        ----------
        x   : tf.Tensor
            Input of shape (batch_size, set_size, input_dim)
        Returns
        -------
        out : tf.Tensor
            Output of shape (batch_size, summary_dim * num_seeds)
        """

        out = self.attention_blocks(x, **kwargs)
        out = self.pooler(out, **kwargs)
        return out


# In[13]:


summary_net = SetTransformer(input_dim=2, attention_settings=dict(num_heads=4, key_dim=32), dense_settings=dict(units=32, activation='relu'))
inference_net = InvertibleNetwork(bf_meta)
amortizer = SingleModelAmortizer(inference_net, summary_net)


# We connect the prior and simulator through a *GenerativeModel* class which will take care of forward inference.

# In[14]:


generative_model = GenerativeModel(prior, batch_simulator)


# In[15]:


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=2000,
    decay_rate=0.95,
    staircase=True,
)


# In[16]:


trainer = ParameterEstimationTrainer(
    network=amortizer, 
    generative_model=generative_model,
    learning_rate = lr_schedule,
    checkpoint_path = './CR3_timelabels_5ACB_[64,64,64]_transformer(8)_ckpts',
    max_to_keep=300,
    skip_checks=True
)


# ### Online training

# In[ ]:


losses = trainer.train_online(epochs=300, iterations_per_epoch=1000, batch_size=128, n_obs=n_obs)
print(losses)
