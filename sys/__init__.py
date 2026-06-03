
from .config import Config, get_config, set_config

__version__ = "1.0.0"

__all__ = [
    'Config',
    'get_config',
    'set_config',
]


def __getattr__(name):
    if name == 'System':
        from .main import System
        return System
    elif name == 'create_system':
        from .main import create_system
        return create_system
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")