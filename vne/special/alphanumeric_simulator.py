from vne.simulate import CHARS, set_default_font, create_example
import matplotlib.pyplot as plt

set_default_font()
plt.rcParams.update(
    {
        "font.size": 22,
    }
)

class alpha_num_Simulator:
    def __init__(self, molecules:str = "aebdijkz2uv"):
        self.molecules = molecules
        
    def keys(self):
        return list(self.molecules)
    
    def __call__(self, key, transform_euler_angles: float = 0, **kwargs):
        # __call__ :  enables for the instances of a class to behave like functions 

        assert key in self.keys() , 'molecule is not in MolecularSimulator'
        return (create_example(key, transform_euler_angles) - 128.) / 128.
    
    def __len__(self):
        return len(self.keys())
