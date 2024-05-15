import os
# import dill
import warnings
import configparser
import numpy as np
import pykoopman as pk
from plot import plot_comparison
from fitK import pkFitK
from step_predictions import step_predictions

# suppress warnings
warnings.filterwarnings("ignore")

model_name = "VanderPol" # CHANGE THIS
## read from config file
config = configparser.ConfigParser()
config.read("config.cfg")
saving = True
plotting = True

# load model and parameters
model_cfg = config[model_name]
data_name = model_cfg["DataPath"]
N_x = int(model_cfg["Nx"])
N_u = int(model_cfg["Nu"])
N_d = int(model_cfg["Nd"])
t0 = float(model_cfg["t0"])
th = float(model_cfg["th"])
obs_tags = model_cfg["obs"].split(", ")

all_cfg = config["ALL"]
degrees = [int(a) for a in all_cfg["degree"].split(", ")]
n_centerss = [int(a) for a in all_cfg["n_centers"].split(", ")]
rand_centers = bool(int(all_cfg["rand_centers"]))
rbf_type = "gauss"
kernel_width = 1.0

current_dir = os.getcwd()
target_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
path = os.path.join(target_dir, "Data", data_name + ".npy")
Xfull = np.load(path)

# compute other parameters
tf = th * (Xfull.shape[1] - 1)
N_xud = [N_x, N_u, N_d]
t_arr = np.arange(t0, tf + th, th)
n_plots, N_T = (
    24,
    Xfull.shape[1],
)  # Number of trajectories to plot, Length of ea. Trajectory

## separate train and test data
train_p = 0.6
idxs = np.arange(Xfull.shape[2])
np.random.shuffle(idxs)
plot_idxs = idxs[:n_plots]

num_train_samples = int(Xfull.shape[2] * (train_p))
train_ix = idxs[:num_train_samples]
test_ix = idxs[num_train_samples:]

# print(Xfull.shape)
print(f"\n\nTraining {model_name} with {obs_tags}...")
for obs_tag in obs_tags:
    dim_itr = degrees if obs_tag == "poly" else n_centerss if obs_tag == "rbf" else [""]
    for dr in dim_itr:

        if rand_centers or obs_tag == "poly":
            centers = None
        else:
            # hi, lo = 3, -3
            # X, Y = np.mgrid[lo:hi+1e-5:(hi-lo)/(np.sqrt(dr)-1), lo:hi+1e-5:(hi-lo)/(np.sqrt(dr)-1)]
            # centers = np.vstack([X.ravel(), Y.ravel()])
            # assert centers.shape[1] == dr
            # kernel_width = (hi-lo)/(np.sqrt(dr)-1)
            if dr == 3:
                centers = np.array([[-1, 1], [1, -1], [0, 0]]).T
            elif dr == 5:
                centers = np.array([[-1, 1], [1, -1], [-1, 1], [1, -1], [0, 0]]).T
            elif dr == 9:
                centers = np.array([[-1, 1], [1, -1], [-1, 1], [1, -1], [-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]]).T
            centers = 2 * centers
            kernel_width = 3.
            print("built grid centers")
        
        koopman_model = pkFitK(Xfull, train_ix, obs_tag, N_xud, centers=centers, rbf_type=rbf_type, n_centers=dr, kernel_width=kernel_width, degree=dr)
        print("Model Type:   ", type(koopman_model))

        # step_predictions(model, Xfull, saving=saving, printing=True, load=False, N_x=N_x, N_u=N_u)
        
        if saving:
            path = os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)
            if obs_tag == "poly":
                koopman_model_name = f"{model_name}_{obs_tag}_deg{dr}"
            elif obs_tag == "rbf": 
                koopman_model_name = f"{model_name}_{obs_tag}_{rbf_type}_nc{dr}_kw{str(kernel_width).replace('.', 'p')}_small"
            
            path = os.path.join(path, "gen", "models", f"{koopman_model_name}.npy")

            print(f"\nSaving to:   {path}")
            # with open(path, "wb") as file:
            np.save(path, koopman_model, allow_pickle=True)

        if plotting:
            ## Compare True and EDMDc
            plot_comparison(koopman_model_name, obs_tag, Xfull, N_T, t_arr, N_xud, plot_idxs, koopman_model, train_ix, saving=saving)

    print(f"\nCompleted.\n")
    # save in gen/models/{model_name}/{obs_tag}/ using dill and name it {model_name}_{obs_tag}_{obs_params}.dill
