import numpy as np
import pykoopman as pk


def pkFitK(X_full, train_ix, obs_tag, N_xud, N_T=100, svd_rank=None, centers=None, rbf_type="gauss", n_centers=10, kernel_width=1.0, degree=3, method="EDMDc"):
    N_x, N_u, N_d = N_xud
    Xb = X_full[:, :N_T, train_ix]

    ## Flatten Data
    Xks, Xkps, Uks = [], [], []
    for i in range(len(train_ix)):
        Xks.append(Xb[:N_x, :-1, i])
        Xkps.append(Xb[:N_x, 1:, i])
        Uks.append(Xb[N_x : N_x + N_u, :-1, i])
    Xk, Xkp, Uk = np.hstack(Xks).T, np.hstack(Xkps).T, np.hstack(Uks).T
    # print(Xk.shape, Xkp.shape, Uk.shape)

    DMDc = pk.regression.DMDc(svd_rank=svd_rank)
    EDMDc = pk.regression.EDMDc()

    # observables
    poly = pk.observables.Polynomial(degree=degree, include_bias=True)
    rbf = pk.observables.RadialBasisFunction(
        rbf_type=rbf_type,
        n_centers=n_centers,
        centers=centers,
        kernel_width=kernel_width,
        polyharmonic_coeff=1.0,
        include_state=True,
    )

    if obs_tag == "poly":
        observable = poly
    elif obs_tag == "rbf":
        observable = rbf
    # fit models
    model = pk.Koopman(observables=observable, regressor=EDMDc).fit(Xk, u=Uk, y=Xkp)
    print(f"\n{obs_tag}: ", model.A.shape)
    # model = pk.Koopman(observables=observables, regressor=DMDc).fit(X0, u=U, y=X1)

    return model


# save the model.
