# 2. manually cutting repeated parts of trajectories

import numpy as np
import matplotlib.pyplot as plt
import os

def find_npz_files(directory):
    npz_files = []
    npz_dirs = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            npz_dirs.append(dir)
        for file in files:
            if (file.endswith(".npz")):
                full_path = os.path.join(root, file)
                npz_files.append(full_path)
    return npz_files, npz_dirs

def plot(X_new, l, r):
    indices = list(range(X_new.shape[0]))
    X_new = X_new[l:r, :]
    indices = indices[l:r]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_new[:, 1], X_new[:, 2], X_new[:, 3], c=indices, cmap='viridis', label='Trajectory Points')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory Plot')

    cbar = plt.colorbar(ax.scatter(X_new[:, 1], X_new[:, 2], X_new[:, 3], c=indices, cmap='viridis'))
    cbar.set_label('Index (i)')

    # plt.savefig(f"{c}.png")
    plt.show()

def read_npz(file_path, l, r):
    data = np.load(file_path)
    plot(data['X'], l, r)
    # np.savez(f"np_data_new/UX_4.npz", X=data['X'][l:r, :], U=data['U'])
    # print(data['X'].shape)
    # print(data['U'].shape)

if __name__ == "__main__":
        wd = os.getcwd() + '/np_data/'
        npz_file_paths, npz_dirs = find_npz_files(wd)
        read_npz(npz_file_paths[4], 499*3, 998*3)
        # parse_trajectories(npz_file_paths, npz_dirs)
        # read_trajectory(npz_file_paths[0])
        print()