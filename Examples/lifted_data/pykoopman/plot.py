import matplotlib.pyplot as plt
import numpy as np
import warnings

import os

warnings.filterwarnings("ignore")


def plot_comparison(
    model_name,
    obs_tag,
    Xfull,
    N_T,
    t_arr,
    N_xud,
    plot_idxs,
    model,
    train_ix,
    no_comp=False,
    EDMDc=True,
    saving=True,
):
    N_x, N_u, N_d = N_xud
    n_trajs_col = 2  # number of trajectories per column
    n_columns = n_trajs_col * 4
    n_plots = len(plot_idxs)
    n_rows = (
        n_plots * 4 + n_columns - 1
    ) // n_columns  # Calculate the number of rows needed

    plt.figure(figsize=(15 * n_columns, 8 * n_rows))

    for i in range(n_plots):
        X = Xfull[:, :N_T, plot_idxs[i]]

        # Calculate positions for the subplots
        pos1 = i * 4 % (n_rows * n_columns) + 1
        pos2 = (i * 4 + 1) % (n_rows * n_columns) + 1
        pos3 = (i * 4 + 2) % (n_rows * n_columns) + 1
        pos4 = (i * 4 + 3) % (n_rows * n_columns) + 1

        ax1 = plt.subplot(n_rows, n_columns, pos1)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.plot(X[0, :], X[1, :], alpha=0.8, label="Nominal", color="blue")

        ax2 = plt.subplot(n_rows, n_columns, pos2)
        ax2.set_xlabel("t")
        ax2.set_ylabel("V")
        ax2.plot(t_arr, X[2, :], alpha=0.8, label="Nominal", color="blue")

        ax3 = plt.subplot(n_rows, n_columns, pos3)
        ax3.set_xlabel("t")
        ax3.set_ylabel("Theta")
        ax3.plot(t_arr, X[3, :], alpha=0.8, label="Nominal", color="blue")

        ax4 = plt.subplot(n_rows, n_columns, pos4)
        ax4.set_xlabel("t")
        ax4.set_ylabel("Error")

        if not (no_comp):
            if not (EDMDc):
                Xhp = model.simulate(
                    X[:N_x, 0].T, X[N_x : N_x + N_u, :].T, n_steps=N_T - 1
                )
                Xh = np.vstack((X[:N_x, 0].T, Xhp)).T
            else:
                # print(X[:N_x, 0].reshape(1, -1).T, X[N_x:N_x+N_u, :].T.shape)
                # print(X[:N_x, 0], X[:N_x, 0].shape, X[N_x:N_x+N_u, :].T.shape)
                # print(i)
                Xh = model.simulate(
                    X[:N_x, 0].reshape(1, -1), X[N_x : N_x + N_u, :].T, n_steps=N_T
                ).T

        ax1.plot(Xh[0, :], Xh[1, :], alpha=0.8, label="Model", color="red")
        # ax2.plot(t_arr, Xh[2, :], alpha=0.8, label="Model", color="red")
        # ax3.plot(t_arr, Xh[3, :], alpha=0.8, label="Model", color="red")
        ax4.plot(
            t_arr,
            np.linalg.norm(X[:N_x, :] - Xh, axis=0),
            alpha=0.8,
            label="Error",
            color="red",
        )

        # Adjust this if you have additional logic for highlighting non-training data
        for ax in [ax1, ax2, ax3, ax4]:
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)  # Increase the width of the plot spines
                if i not in train_ix:
                    spine.set_edgecolor("red")
    plt.subplots_adjust(
        left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4
    )
    if no_comp:
        plt.suptitle(f"{model_name}'s Trajectories", fontsize=20)
    else:
        plt.suptitle(
            f"{model.regressor} Traj. Linearizations ({obs_tag}) of {model_name}",
            fontsize=20,
        )
    # plt.show()
    if saving:
        path = os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)
        path = os.path.join(path, "gen", "plots", f"{model_name}")
        # print(path)
        # with open(path, "wb") as file:
        plt.savefig(path)
    # save plot in gen/plots/{model_name}. name is {model_name}_{obs_tag}_{obs_params}


# load model from saved location and generate based on that.
