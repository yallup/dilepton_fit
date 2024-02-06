import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("events.csv", header=None)
x = df[1].to_numpy()
y = df[3].to_numpy()
yerr = np.sqrt(y)


def breit_w(x, mass=91.1876, width=2.4952):
    bw = width / ((x - mass) ** 2 * width**2 / 4)
    return bw


def dilepton_fit(theta, c=1):
    bw = breit_w(x)
    # bw = 1
    f = (
        (1 - (x / 13000) ** c) ** theta[..., 1][..., None]
        * (x / 13000)
        ** np.sum(
            theta[..., 2:][..., None]
            * np.log(x / 13000) ** np.arange(theta.shape[-1] - 2)[..., None],
            axis=-2,
        )
        * bw
    )
    norm = theta[..., 0] / f.sum(axis=-1)
    return norm[..., None] * f


theta = np.array([1.78e5, 1.5, -12.38, -4.295, -0.9191, -0.0845])

## Can add an extra parameter to the power and it will roll over that dimension, but this needs to be fit for properly
# theta = np.array([1.78e5, 1.5, -12.38, -4.295, -0.9191, -0.0845,-0.0019])

y_pred = dilepton_fit(theta)


f, a = plt.subplots(
    nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
f.subplots_adjust(hspace=0)
# a[0].plot(x, theta[0] * breit_w(x)/breit_w(x).sum(), label="Breit-Wigner")
a[0].errorbar(x, y, yerr=yerr, color="black", fmt="None")
a[0].scatter(x, y, color="black", s=6, label="Data")
a[0].plot(x, y_pred, c="red", label="Prediction")
a[0].legend()
a[0].set_ylim(1e-2, 3e4)
a[0].loglog()
a[0].set_xlim(x.min(), x.max())

a[1].hlines(1, x.min(), x.max(), color="red")
a[1].scatter(x, y / y_pred, color="black", s=6)
a[1].errorbar(x, y / y_pred, yerr=yerr / y_pred, fmt="None", color="black")
a[1].set_ylim(0.25, 1.75)
a[1].set_xlabel("Invariant mass [GeV]")
a[1].set_ylabel("Data/Prediction")
a[0].set_ylabel("Events / 10 GeV")


# We can check this is vectorized by sampling from the prior and plotting the predictions
from scipy.stats import multivariate_normal

prior = multivariate_normal(mean=theta, cov=np.eye(theta.shape[-1]) * 0.0001)
prior_samples = prior.rvs(10)
y_pred_samples = dilepton_fit(prior_samples)
a[0].plot(
    np.repeat(x.reshape(-1, 1), 10, axis=1),
    y_pred_samples.T,
    c="blue",
    alpha=0.5,
)
a[1].plot(
    np.repeat(x.reshape(-1, 1), 10, axis=1),
    (y_pred_samples / y_pred).T,
    c="blue",
    alpha=0.5,
)

f.savefig("dielectron_paper_fit.pdf", bbox_inches="tight")
