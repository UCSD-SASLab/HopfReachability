import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def step_predictions(model_name, data_name, saving=True, printing=False, load=True, N_x=4, N_u=2):

    if load:
        model_npy = np.load(f"gen/models/{model_name}.npy", allow_pickle=True)
        model = model_npy.item()
        data = np.load(f"data/{data_name}.npy", allow_pickle=True)
    else:
        model = model_name
        data = data_name
    # print(type(model))
    # print(data.shape)  # (5, 101, 2000)

    Xfull = data
    N_T = 46
    # N_x = 2  # TODO: read from config
    # N_u = 2

    s1_norms_all = []
    s5_norms_all = []
    s10_norms_all = []

    for i in range(data.shape[2]):  # data.shape[2]
        s1_norms = []
        s5_norms = []
        s10_norms = []
        for j in range(data.shape[1] - 1):  # data.shape[1] - 1
            x1 = Xfull[:N_x, j, i]
            u = Xfull[N_x : N_x + N_u, j, i]
            x2_true = Xfull[:N_x, j + 1, i]

            # 1 step
            x2_sim = model.predict(x1.reshape(1, -1), u.reshape(1, -1))
            # print(x2_sim, x2_true)
            x2_err = np.linalg.norm(x2_sim - x2_true)  # 2-norm of x2_sim and x2_true
            # print(x2_err)
            s1_norms.append(x2_err)

            for ts in range(1, 10, 1):
                if j + ts < N_T:
                    u_ts = Xfull[N_x : N_x + N_u, j + ts, i]
                    x2_sim = model.predict(x2_sim, u_ts.reshape(1, -1))
                    if ts == 4 and (j + 5 < N_T):  # time step 5 has been simulated
                        x2_true = Xfull[:N_x, j + 5, i]
                        x2_err = np.linalg.norm(
                            x2_sim - x2_true
                        )  # 2-norm of x2_sim and x2_true
                        s5_norms.append(x2_err)
                    if ts == 9 and (j + 10 < N_T):  # time step 10 has been simulated
                        x2_true = Xfull[:N_x, j + 10, i]
                        x2_err = np.linalg.norm(
                            x2_sim - x2_true
                        )  # 2-norm of x2_sim and x2_true
                        s10_norms.append(x2_err)
        s1_norms_all.append(s1_norms)
        s5_norms_all.append(s5_norms)
        s10_norms_all.append(s10_norms)

    s1_norms_all = np.array(s1_norms_all)
    s5_norms_all = np.array(s5_norms_all)
    s10_norms_all = np.array(s10_norms_all)

    if printing:
        print("Mean & Std of 1-step  error =", round(np.mean(s1_norms_all), 3), round(np.std(s1_norms_all), 3))
        print("Mean & Std of 5-step  error =", round(np.mean(s5_norms_all), 3), round(np.std(s5_norms_all), 3))
        print("Mean & Std of 10-step error =", round(np.mean(s10_norms_all), 3), round(np.std(s10_norms_all), 3))

    if saving:
        base_path = f"gen/n_step_norms/{model_name}"

        if not os.path.exists(base_path):
            os.makedirs(base_path)
            
        np.save(os.path.join(base_path, "s1_norms_all.npy"), s1_norms_all)
        np.save(os.path.join(base_path, "s5_norms_all.npy"), s5_norms_all)
        np.save(os.path.join(base_path, "s10_norms_all.npy"), s10_norms_all)


# step_predictions("Sphero_poly_3", "SpheroData")
# step_predictions("Sphero_rbf", "SpheroData")

# step_predictions("Dubins_poly_7", "Dubins_data_tf5_2k_all")
