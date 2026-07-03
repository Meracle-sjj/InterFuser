import numpy as np

# Fix imgaug compatibility with NumPy 2.0
# NumPy 2.0 removed np.sctypes, so we patch it for imgaug
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'bool_': [np.bool_],
        'int_': [np.int8, np.int16, np.int32, np.int64],
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float32, np.float64],
        'float_': [np.float32, np.float64],
        'complex_': [np.complex64, np.complex128],
        'complex': [np.complex64, np.complex128],
        'object_': [np.object_],
    }

import imgaug as ia
from imgaug import augmenters as iaa

def augment(prob=0.2):
    augmenter = iaa.Sequential([
        iaa.Sometimes(prob, iaa.GaussianBlur((0, 0.5))),
        iaa.Sometimes(prob, iaa.AdditiveGaussianNoise(loc=0, scale=(0., 0.05*255), per_channel=0.5)),
        iaa.Sometimes(prob, iaa.Dropout((0.01, 0.1), per_channel=0.5)),
        iaa.Sometimes(prob, iaa.Multiply((1/1.2, 1.2), per_channel=0.5)),
        iaa.Sometimes(prob, iaa.LinearContrast((1/1.2, 1.2), per_channel=0.5)),
        iaa.Sometimes(prob, iaa.Grayscale((0.0, 1))),
        iaa.Sometimes(prob, iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
    ], random_order=True)
    return augmenter

