import os
import gemmi
import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP
from vne.metrics import similarity

class Protein_PDB:
    """ 
    This class reads *.cif file and creates a Protein objects 

    Parameters
    ----------
    filename : string
        path+name of the *.cif file 

    Axes : list
        the coordinate axes  
    doc : 
        all the information from the cif file read by Gemmi
    block : 
        The cif file in a single block
    positions : 
        All the positions of the atoms in the protein
    symbols : 
        the corresponding symbols 
    species : 
        List of the unique species in the protein
    """
    def __init__(self, 
                filename):
        self.AXES = ["Cartn_x", "Cartn_y", "Cartn_z"]
        doc = gemmi.cif.read_file(filename)  # copy all the data from mmCIF file
        block = doc.sole_block()  # mmCIF has exactly one block
        self.positions = block.find("_atom_site.", self.AXES)
        self.symbols = list(block.find_values("_atom_site.type_symbol"))
        self.species = list(block.find_loop('_atom_type.symbol'))

        self._preprocess()
        self._choose_symbols()
            

    def _preprocess(self):
        """ Pre-process the symbols and species
            This function corrects the symbols and species so that 
            only the the first letter is Capitalised 
                * converts "MG" -> to "Mg"
        """

        self.symbols = [s.title() for s in self.symbols]
        self.species = [s.title() for s in self.species]
        
    def _choose_symbols(self):
        """
        Number of species 
        """
        elements =  ['C', 'N', 'O']
        self.symbols = [ x if x in elements else 'C' for x in self.symbols ]
        self.species = [ x if x in elements else 'C' for x in self.symbols ]
        

def pdb_coord(protein : Protein_PDB)-> np.ndarray:
    
    positions = protein.positions
    coords = np.stack(
        [
            [float(r) for r in positions.column(idx)]
            for idx in range(len(protein.AXES))
        ],
        axis = -1
    )
    # center the molecule in XYZ
    # centroids = np.mean(coords, axis=0)
    # coords = coords - centroids

    return coords

def pdb_centre(coords)-> np.ndarray:
    # find the centre of the protein for SOAP calculation
    
    protein_centre = []
    for i in range(3):
        protein_centre.append(coords[:,i].min() + (coords[:,i].max()- coords[:,i].min())/2)
    
    dist = []
    
     
    # print(centroids - protein_centre)
    dist_vect = coords.copy()
    
    for i in range(len(coords)):
        dist_vect[i,:] = coords[i, :] - protein_centre[:]
        dist.append(abs(np.sqrt(np.sum(dist_vect[i,:]**2))))

    dist_index = np.flip(np.argsort(dist))
    centre_index = np.flip(np.argsort(dist))[-1]

    dist = np.sort(dist)

    new_coords = [coords[i] for i in dist_index]

    return new_coords

def pdb_features(protein : Protein_PDB) -> np.array:
    coords = pdb_coord(protein)
    positions = coords.tolist()
    atoms = Atoms(symbols=protein.symbols, positions=positions)
    
    # shift the centre to [0,0,0]
    coords = pdb_centre(coords)
    centre = [-1,-1,-1]

    rcut = 6.0
    nmax = 8
    lmax = 6
    # Setting up the SOAP descriptor
    soap = SOAP(
        species=protein.species,
        periodic=False,
        rcut=rcut,
        nmax=nmax,
        lmax=lmax,
    )
    
    features = soap.create(
        atoms,
        positions=centre,
    )
    return features  # normalize(features)

def similarity_matrix(molecules):
    n_molecules = len(molecules)
    m = []
    for idx, c_i in enumerate(molecules):
        f_i = pdb_features(c_i)
        for jdx, c_j in enumerate(molecules):
            f_j = pdb_features(c_j)
            m.append(similarity(f_i, f_j))
    return np.reshape(m, (n_molecules, n_molecules))



