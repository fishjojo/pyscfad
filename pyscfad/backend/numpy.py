from .config import get_backend

def __getattr__(name):
    return getattr(get_backend(), name)
