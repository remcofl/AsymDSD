from jsonargparse import lazy_instance

from ..components import *

DEFAULT_CLASSIFIER_OPTIMIZER = lazy_instance(SGDSpec, weight_decay=0.0)
