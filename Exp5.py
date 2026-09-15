import numpy as np
import matplotlib.pyplot as plt

def lowess(x, y, f):
    n = len(x)
    r = int(f * n)
    yest = np.zeros(n)

    for i in range(n):
        # Calculate distance from current point
        d = abs(x - x[i])

        # Select nearby points
        idx = np.argsort(d)[:r]

        # Calculate weights
        w = (1 - (d[idx] / d[idx[-1]])**3)**3

        # Weighted linear regression
        X = np.column_stack((np.ones(len(idx)), x[idx]))

        beta = np.linalg.lstsq(X * w[:, None], y[idx] * w, rcond=None)[0]

        yest[i] = beta[0] + beta[1] * x[i]

    return yest


# Generate data
n = 100
x = np.linspace(0, 2*np.pi, n)
y = np.sin(x) + 0.3*np.random.randn(n)

# Apply LOWESS
yest = lowess(x, y, 0.25)

# Plot
plt.plot(x, y, "r.")
plt.plot(x, yest, "b-")
plt.show()