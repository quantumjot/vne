import numpy as np
import matplotlib.pyplot as plt
import umap

def plot_affinity(lookup,molecule_list):

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(lookup, vmin=-1, vmax=1, cmap=plt.cm.get_cmap('RdBu'))
    ax.set_title('Shape similarity matrix')
    ax.set_xticks(np.arange(0, len(molecule_list)))
    ax.set_xticklabels(molecule_list)
    ax.set_yticks(np.arange(0, len(molecule_list)))
    ax.set_yticklabels(molecule_list)
    ax.tick_params(axis='x', rotation=90)
    fig.colorbar(im, ax=ax)
    plt.savefig('similarity.png', dpi=144)



def plot_loss(loss_plot,  kldloss_plot,sloss_plot, rloss_plot):
    plt.figure(figsize=(16,16))
    plt.plot(loss_plot, linewidth=3)
    plt.plot(kldloss_plot, linewidth=3)
    plt.plot(sloss_plot, linewidth=3)
    plt.plot(rloss_plot, linewidth=3)

    plt.xlabel('EPOCHS')
    plt.ylabel('Total loss')
    plt.savefig("loss_vs_EPOCHS.png", dpi=144)


def plot_umap(enc,lbl,epoch,  molecule_list):
    enc = np.concatenate(enc, axis=0)

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(enc)


    fig, ax = plt.subplots(figsize=(16, 16))
    for mol_id, mol in enumerate(molecule_list):
        idx = np.where(np.array(lbl) == mol_id)[0]
        cmap = plt.cm.get_cmap("tab20")
        color = cmap(mol_id % 20)
        
        scatter = ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            s = 64,
            label = mol[:4],
            facecolor=color,
            edgecolor=color,
        )
        
    #     ax.plot(x_enc[:, 0], x_enc[:, 1], 'ko', markersize=42)
    ax.legend()
    ax.set_title(f'UMAP projection', fontsize=24)
    plt.savefig(f'UMAP{epoch}.png', dpi=144)


def plot_pose(model):
    pose_angle = []
    r = []

    with torch.inference_mode(model):
        for theta in range(-45,45):
            y = dataset[3]
            x, z, z_pose, mu, logvar = model(y[0][np.newaxis, ...].to(device=device))     
            if theta % 5 == 0 : 
                r.append(np.squeeze(x[0,0,...].cpu()))
            pose_angle.append(z_pose[0,0].cpu().clone().numpy())

    fig, ax = plt.subplots(2,1, figsize=(10,15))
    ax1, ax2 = ax


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1) #. Clamps all elements in input into the range [ min, max ]. 
    x = x.view(x.size(0), 1, 64, 64) # Returns a new tensor with the same data as the self tensor but of a different shape.
    return x
