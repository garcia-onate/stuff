"""Configuration loading utilities.

Load and parse YAML configuration files.
"""

import yaml
import os


def load_config(config_path=None):
    """Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str, optional
        Path to config file. If None, loads default config.
        
    Returns
    -------
    dict
        Configuration dictionary
        
    Examples
    --------
    >>> config = load_config()
    >>> config = load_config('my_config.yaml')
    """
    if config_path is None:
        # Load default config
        config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'configs')
        config_path = os.path.join(config_dir, 'default_config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_nested(config, keys, default=None):
    """Get nested value from config dict.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    keys : str or list
        Dot-separated key string or list of keys
    default : any, optional
        Default value if key not found
        
    Returns
    -------
    any
        Configuration value or default
        
    Examples
    --------
    >>> value = get_nested(config, 'training.learning_rate')
    >>> value = get_nested(config, ['training', 'learning_rate'])
    """
    if isinstance(keys, str):
        keys = keys.split('.')
    
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value
