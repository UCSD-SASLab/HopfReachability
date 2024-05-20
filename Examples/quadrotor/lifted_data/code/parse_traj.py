# 3. combining controls and states

import numpy as np
import matplotlib.pyplot as plt
import os


X_pre = []
U_mid = []
X_post = []

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

# combining controls and states
def combine(file_path, u_buffer=5, x_buffer=1):
    data = np.load(file_path)
    l = 0; r = data['X'].shape[0]

    X = data['X']
    U = data['U']

    results = []
    u_idx = 0
    # for i, x in enumerate(X[:-u_buffer]):
    for i in range(0, X.shape[0] - x_buffer, x_buffer):
        # find all U.time > X.time
        while u_idx < len(U) and U[u_idx][0] <= X[i, 0]:
            u_idx += 1
        
        u = U[u_idx : min(u_idx+u_buffer, U.shape[0])]
        u = [j for sub in u for j in sub[1:]]
        u = np.array(u)

        x1 = np.array(X[i])
        x2 = np.array(X[i+x_buffer])
        xux = np.concatenate((x1, u, x2))
        X_pre.append(x1)
        U_mid.append(u)
        X_post.append(x2)
        results.append(xux)
        # get x+ and x- from here !!

    # np.savez(f"np_data_final/UX_0.npz", X=data['X'][l:r, :], U=data['U'])
    return np.array(results) # shape[1] = len(x) + len(u) * ctrl_buffer_max

def combine_ordered(file_path, u_buffer=5, x_buffer=1):
    data = np.load(file_path)
    l = 0; r = data['X'].shape[0]
    # print(l, r)

    X = data['X']
    print("a", X.shape)
    U = data['U']

    results = []
    u_idx = 0
    # for i, x in enumerate(X[:-u_buffer]):
    X1_cat = []
    U_cat = []
    X2_cat = []
    x1 = None
    x2 = None
    for i in range(l, r - x_buffer, x_buffer):
        # find all U.time > X.time
        while u_idx < len(U) and U[u_idx][0] <= X[i, 0]:
            u_idx += 1
        u = U[u_idx : min(u_idx+u_buffer, U.shape[0])]
        u = [j for sub in u for j in sub[1:]]
        u = np.array(u)

        x1 = np.array(X[i])
        # if not np.array_equal(x1, x2): print('a', i)
        x2 = np.array(X[i+x_buffer])
        xux = np.concatenate((x1, u, x2))
        X1_cat.append(x1)
        U_cat.append(u)
        X2_cat.append(x2)
        results.append(xux)
        # get x+ and x- from here !!

    # np.savez(f"np_data_final/UX_0.npz", X=data['X'][l:r, :], U=data['U'])
    return X1_cat, U_cat, X2_cat # shape[1] = len(x) + len(u) * ctrl_buffer_max


# if __name__ == "__main__":
#         wd = os.getcwd() + '/np_data_new/'
#         npz_file_paths, npz_dirs = find_npz_files(wd)
#         combined_results = None
#         x_buffer, u_buffer = 3, 2
#         for i, path in enumerate(npz_file_paths):
#             results = combine(npz_file_paths[i], u_buffer=u_buffer, x_buffer=x_buffer)
#             if combined_results is None:
#                 combined_results = results
#             else:
#                 combined_results = np.concatenate((combined_results, results), axis=0)
#         combined_results = np.array(combined_results)
#         print(combined_results.shape)
#         np.savez(f"np_data_final/traj_div__x{x_buffer}_u{u_buffer}", X0=X_pre, U=U_mid, X1=X_post)
#         print()


if __name__ == "__main__":
    wd = os.getcwd() + '/np_data_new/'
    npz_file_paths, npz_dirs = find_npz_files(wd)
    X_pre, U_mid, X_post = [], [], []
    x_buffer, u_buffer = 3, 2

    for i, path in enumerate(npz_file_paths):
        X1_cat, U_cat, X2_cat = combine_ordered(path, u_buffer=u_buffer, x_buffer=x_buffer)
        X_pre.append(X1_cat)
        U_mid.append(U_cat)
        X_post.append(X2_cat)

        # if (i == 3):
        #     X1_array_t = np.array(X1_cat)
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.plot(X1_array_t[:, 1], X1_array_t[:, 2], X1_array_t[:, 3]) # looks fine
        #     print(X1_array_t.shape)
        #     plt.show()
    
    # print("a", len(X_pre[1]))

    # Calculate the minimum length of trajectories
    min_len = min(min(len(x) for x in X_pre), min(len(u) for u in U_mid), min(len(x) for x in X_post))

    # Splitting trajectories
    X_pre_split, U_mid_split, X_post_split = [], [], []

    for x, u, x2 in zip(X_pre, U_mid, X_post):
        print(len(x)); # 999
        # Number of full segments that can fit in the trajectory
        num_segments = len(x) // min_len
        # print(len(x), num_segments)
        for segment in range(num_segments):
            start_idx = segment * min_len
            end_idx = start_idx + min_len
            print(start_idx, end_idx)
            X_pre_split.append(x[start_idx:end_idx])
            U_mid_split.append(u[start_idx:end_idx])
            X_post_split.append(x2[start_idx:end_idx])
        print()
    
    X_pre_split = np.array(X_pre_split)
    U_mid_split = np.array(U_mid_split)
    X_post_split = np.array(X_post_split)


    print(X_pre_split.shape)
    print(U_mid_split.shape)
    print(X_post_split.shape)

    # np.savez(f"np_data_final/traj_div2__x{x_buffer}_u{u_buffer}", X0=X_pre_split, U=U_mid_split, X1=X_post_split)