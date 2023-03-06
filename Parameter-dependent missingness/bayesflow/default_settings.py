from abc import ABC, abstractmethod


class Setting(ABC):
    """Abstract Base class for settings. It's here to potentially extend the setting functionality in future.
    """
    @abstractmethod
    def __init__(self):
        """"""
        pass


class MetaDictSetting(Setting):
    """Implements an interface for a default meta_dict with optional mandatory fields
    """
    def __init__(self, meta_dict: dict, mandatory_fields: list = []):
        """

        Parameters
        ----------
        meta_dict: dict
            Default dictionary.

        mandatory_fields: list, default: []
            List of keys in `meta_dict` that need to be provided by the user.
        """
        self.meta_dict = meta_dict
        self.mandatory_fields = mandatory_fields


DEFAULT_SETTING_INVARIANT_BAYES_FLOW = MetaDictSetting(
    meta_dict={
            'n_coupling_layers': 4,
            's_args': {
                'n_dense_h1': 2,
                'n_dense_h2': 3,
                'dense_h1_args': {'units': 64, 'activation': 'relu', 'kernel_initializer': 'glorot_uniform'},
                'dense_h2_args': {'units': 128, 'activation': 'relu', 'kernel_initializer': 'glorot_uniform'},
            },
            't_args': {
                'n_dense_h1': 2,
                'n_dense_h2': 3,
                'dense_h1_args': {'units': 64, 'activation': 'relu', 'kernel_initializer': 'glorot_uniform'},
                'dense_h2_args': {'units': 128, 'activation': 'relu', 'kernel_initializer': 'glorot_uniform'},
            },

            'alpha': 1.85,
            'permute': True
        },
    mandatory_fields=['n_params', 'n_models']
)

DEFAULT_SETTING_INVARIANT_NET = MetaDictSetting(
    meta_dict={
        'n_dense_s1': 2,
        'n_dense_s2': 2,
        'n_dense_s3': 2,
        'n_equiv':    2,
        'dense_s1_args': {'activation': 'relu', 'units': 32},
        'dense_s2_args': {'activation': 'relu', 'units': 64},
        'dense_s3_args': {'activation': 'relu', 'units': 32}
    },
    mandatory_fields=[]
)


DEFAULT_SETTING_INVERTIBLE_NET = MetaDictSetting(
    meta_dict={
        'n_coupling_layers': 4,   # default: 4
        's_args': {
            'units': [128, 128],  # default: [128, 128]
            'activation': 'elu',
            'initializer': 'glorot_uniform',
        },
        't_args': {
            'units': [128, 128],  # default: [128, 128]
            'activation': 'elu',
            'initializer': 'glorot_uniform',
        },
        'alpha': 1.9,
        'use_permutation': True,
        'use_act_norm': True,
        'act_norm_init': None,
        'adaptive_tails': False,
        'tail_network': {}
    },
    mandatory_fields=["n_params"]
)

MMD_BANDWIDTH_LIST = [
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
    1e3, 1e4, 1e5, 1e6
]