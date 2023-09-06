import sys
import matplotlib.pyplot as plt

from vne.vae import ShapeVAE, ShapeSimilarityLoss
from vne.special.affinity_mat_create import similarity_matrix
from vne.special.alphanumeric_simulator import  alpha_num_Simulator
from vne.vis import plot_affinity

alpha_num_list = "aebdijkz2uv"
aff_mat = None
data_nat = 'alphanum'

if aff_mat is None and data_nat == 'alphanum':
    simulator = alpha_num_Simulator()
    lookup, imgs = similarity_matrix(simulator)



plot_affinity(lookup,simulator.keys())


