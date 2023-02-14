import os

import tensorflow as tf
import matplotlib.pyplot as plt

from model import make_generator_model

if __name__ == "__main__":
    generator = make_generator_model()
