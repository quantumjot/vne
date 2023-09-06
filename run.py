import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
from vne.vae import ShapeVAE, ShapeSimilarityLoss
from vne.special.affinity_mat_create import similarity_matrix
from vne.special.alphanumeric_simulator import  alpha_num_Simulator
from vne.vis import plot_affinity
from vne.dataset import alphanumDataset, SubTomogram_dataset, CustomMNIST
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


alpha_num_list = "aebdijkz2uv"
aff_mat = None
data_nat = 'alphanum'
classes = 'classes.csv'


with open(classes, newline='') as molecule_list_file:
    molecule_list = list(csv.reader(molecule_list_file, delimiter=','))[0]


if aff_mat is None and data_nat == 'alphanum':
    simulator = alpha_num_Simulator()
    lookup, imgs = similarity_matrix(simulator)

elif aff_mat:
    lookup =  genfromtxt(aff_mat, delimiter=',')


plot_affinity(lookup,simulator.keys())



if data_nat == "mnist":
    dataset = CustomMNIST(root='./data', train=True)
    test_dataset = CustomMNIST(root='./data', train=False)

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
elif data_nat=="subtomo":
    dataset = SubTomogram_dataset(subtomo_path,IMAGES_PER_EPOCH, molecule_list)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last= True)
    molecule_list = dataset.keys()
elif data_nat=="alphanum":
    dataset = alphanumDataset(-45,45, list(alpha_num_list), simulator)


x = dataset[0]
fig = plt.figure()
fig.colorbar(plt.imshow(np.squeeze(x[0].numpy())))
#plt.imshow(to_img(x))
fig.savefig('data_before_loss_calc.png', dpi=144)
