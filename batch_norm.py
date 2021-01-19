import tensorflow as tf
import numpy as np


class BatchNorm:

    def __init__(self, is_train, use_batch_norm):
        self.is_train = is_train
        self.use_batch_norm = use_batch_norm


