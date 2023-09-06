from vne.features import image_to_features
from vne.metrics import similarity
import numpy as np

def similarity_matrix(simulator):
    # this function calculates the similarity between all the molecules in the MolecularSimulator 
    n_molecules = len(simulator)
    m = []
    examples = np.zeros((n_molecules, 64, 64))
    
    for idx, c_i in enumerate(simulator.keys()): 
        x_i = simulator(c_i, project=False) >0
        f_i = image_to_features(x_i)

        for c_j in simulator.keys():
            
            x_j = simulator(c_j, project=False) >0 
            f_j = image_to_features(x_j)
            
            m.append(similarity(f_i, f_j))
            
            # why is the sum necessary here ?
            
        examples[idx, ...] =x_i #np.sum(x_i, axis=-1)
            
    return np.reshape(m, (n_molecules, n_molecules)), examples
