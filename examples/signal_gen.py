from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 3000, 100)
N_samples = 1000

# gen prior samples on range mean in [500, 2500] and sigma in [10, 50] and amplitude in [0, 500]
mean = np.random.rand(N_samples) * 2000 + 500
sigma = np.random.rand(N_samples) * 40 + 10
amplitude = np.random.rand(N_samples) * 500

# gen gaussian dists
gaussian_dists = norm(loc=mean[..., None], scale=sigma[..., None]).pdf(x)
# normalise to 1
gaussian_dists /= gaussian_dists.max(axis=-1)[..., None]

# scale by amplitude
gaussian_dists *= amplitude[..., None]


# plot the first 20 signals
f, a = plt.subplots(1, 1)
a.plot(x, gaussian_dists[:20].T, color="C0", alpha=0.3)

# draw the actual discrete "data"
a.scatter(x, gaussian_dists[0])

f.savefig("signal_gen.pdf")


print("Done")
