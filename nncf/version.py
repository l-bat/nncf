from nncf.common.utils.backend import __nncf_backend__

if __nncf_backend__ == 'Torch':
    __version__ = "1.7.1"
    BKC_TORCH_VERSION = "1.8.1"
elif __nncf_backend__ == 'Tensorflow':
    __version__ = "1.0"
