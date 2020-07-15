import json
from .doc3dwc_loader import doc3dwcLoader
from .doc3dbmnoimgc_loader import doc3dbmnoimgcLoader
from .doc3djoint_loader import doc3djointLoader

def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'doc3dwc': doc3dwcLoader,
        'doc3dbmnic': doc3dbmnoimgcLoader,
        'doc3djoint': doc3djointLoader,
    }[name]
