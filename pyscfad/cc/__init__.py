from . import rccsd
from . import dfccsd

def RCCSD(mf, *args, **kwargs):
    return rccsd.RCCSD(mf, *args, **kwargs)
