import gym 
import pybullet_envs
import pybullet as p
import time
import math
import threading

from test2_live_plotter import MyDataClass, MyPlotClass
from complexity_subgoals import OUActionNoise

# it is different from how MuJoCo renders environments
# it doesn't differ too much to me w/ and w/o mode='human'
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
# p.connect(p.DIRECT)