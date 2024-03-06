import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

df = pd.read_csv("events.csv", header=None)
x = df[0].to_numpy()

bw = (df[2] - df[1]).to_numpy()
y = torch.round(torch.tensor(((df[2]-df[1]) * df[3]).to_numpy())/10)
# y_plot= df[3].to_numpy()

yerr = np.sqrt(y.detach())

mask = y > 0.0
# X_tensor = torch.tensor(x[mask], dtype=torch.float32)
# y_tensor = torch.tensor(y[mask])
# y_err_tensor = torch.tensor(yerr[mask])


X_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y)


def breit_w(x, mass=91.1876, width=2.4952):
    bw = width / ((x - mass) ** 2 - width**2 / 4)
    return bw


def dilepton_fit(theta, c=1):
    bw = breit_w(X_tensor)
    # bw = 1
    f = (
        (1 - (X_tensor / 13000) ** c) ** theta[..., 1][..., None]
        * (X_tensor / 13000)
        ** torch.sum(
            theta[..., 2:][..., None]
            * np.log(X_tensor / 13000)
            ** np.arange(theta.shape[-1] - 2)[..., None],
            axis=-2,
        )
        * bw
    )
    norm = theta[..., 0] * 1e5 / f.sum(axis=-1)
    return norm[..., None] * f


theta = np.array([1.78, 1.5, -12.38, -4.295, -0.9191, -0.0845])
# theta = np.array([1.78, 1.5, -12.38, 1.0])

## Can add an extra parameter to the power and it will roll over that dimension, but this needs to be fit for properly
# theta = np.array([1.78e5, 1.5, -12.38, -4.295, -0.9191, -0.0845,-0.0019])

# y_pred = dilepton_fit(theta)


f, a = plt.subplots(
    nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
f.subplots_adjust(hspace=0)
# a[0].plot(x, theta[0] * breit_w(x)/breit_w(x).sum(), label="Breit-Wigner")
a[0].errorbar(x, y, yerr=yerr, color="black", fmt="None")
a[0].scatter(x, y, color="black", s=6, label="Data")
# a[0].plot(x, y_pred, c="red", label="Prediction")
a[0].legend()
a[0].set_ylim(1e-2, 3e4)
a[0].loglog()
a[0].set_xlim(x.min(), x.max())

a[1].hlines(1, x.min(), x.max(), color="red")
# a[1].scatter(x, y / y_pred, color="black", s=6)
# a[1].errorbar(x, y / y_pred, yerr=yerr / y_pred, fmt="None", color="black")
a[1].set_ylim(0.25, 1.75)
a[1].set_xlabel("Invariant mass [GeV]")
a[1].set_ylabel("Data/Prediction")
a[0].set_ylabel("Events / 10 GeV")


# We can check this is vectorized by sampling from the prior and plotting the predictions
from scipy.stats import multivariate_normal

# prior = multivariate_normal(mean=theta, cov=np.eye(theta.shape[-1]) * .01)
# prior = multivariate_normal(mean=np.zeros_like(theta))


class uniform_prior:
    def __init__(self, n):
        self.n = n

    def rvs(self, size=1):
        mod = 10.0 ** -np.arange(self.n - 2)
        cube = np.random.rand(size, self.n) * 10
        cube[..., 1] = cube[..., 1]
        cube[..., 2:] = cube[..., 2:] * -1 * mod

        return cube.squeeze()


prior = uniform_prior(7)

import torch
from torch import nn


def loglike(y_pred, y):
    # define a log likelihood (basically a MSE loss), note in reality this should have the \Sigma included but as we are optimizing we don't care about that for now
    # return torch.mean((y_pred - y) ** 2 / y_err_tensor**2)
    return nn.PoissonNLLLoss(log_input=False, full=True)(y_pred, y)


N_iter = 100000
from torch.optim import Adam

# To write things in a torch-ey way we have to put things into tensors
theta = torch.tensor(prior.rvs(), dtype=torch.float32, requires_grad=True)

# This is overkill for this but we will use the adam optimizer
optimizer = Adam([theta], lr=0.01)

lowest_loss = float('inf')
best_theta = None
best_y = None
losses = []
for i in range(N_iter):
    optimizer.zero_grad()
    y_pred = dilepton_fit(theta)

    loss = loglike(y_pred, y_tensor)

    if loss < lowest_loss:
        lowest_loss = loss
        best_theta = theta.detach().clone()
        best_y= y_pred.detach().clone()
    if i % 10000 == 0:
        a[0].plot(
            X_tensor.detach().numpy(),
            y_pred.detach().numpy(),
            c="red",
            label=f"iter {i}",
            alpha=i / N_iter,
        )
        a[1].plot(
            X_tensor.detach().numpy(),
            y_tensor.detach().numpy() / y_pred.detach().numpy(),
            c="red",
            label=f"iter {i}",
            alpha=i / N_iter,
        )
        print(loss)
    # Backward pass and optimize

    loss.backward()
    optimizer.step()
    losses.append(loss.detach().numpy())

a[0].plot(
    X_tensor.detach().numpy(),
    best_y.numpy(),
    c="C0",
    label=f"iter {i}",
)

a[0].legend()

f_loss, a_loss = plt.subplots()
a_loss.plot(losses)
a_loss.set_yscale("log")
a_loss.set_ylabel("Loss")
a_loss.set_xlabel("Iteration")
f_loss.savefig("loss.pdf", bbox_inches="tight")

print(best_theta)

f.savefig("dielectron_fit.pdf", bbox_inches="tight")
