import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.polynomial.hermite import hermval
import numpy as np

# Font size
font = {'family': 'normal',
        'weight': 'normal',
        'size': 22}

plt.rc('font', **font)

# Initialize the plot
fig = plt.figure()
gs = gridspec.GridSpec(1, 2)
ax1 = fig.add_subplot(gs[0], adjustable='box')
ax2 = fig.add_subplot(gs[1], adjustable='box')
fig.set_size_inches(8, 5)

# Set up axes
ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])
ax2.get_xaxis().set_ticks([])
ax2.get_yaxis().set_ticks([])
ax1.set_xlabel("Position")
ax1.set_ylabel("Potential energy")
ax2.set_xlabel("Position")

# The harmonic oscillator potential
x_min = -3
x_max = 3
x = np.linspace(x_min, x_max, 50)
V = x**2
ax1.plot(x, V, lw=2)
ax1.set_title('Classical')
ax2.plot(x, V, lw=2)
ax2.set_title('Quantum')

# ---- The classical states ----
# Draw the circles
# circle_radius = 0.2
# circleRest_center = [0, 0 + circle_radius]
# circleRest = plt.Circle((0, 0.2), circle_radius, color='r')
# circleFast_center = (1.25, 1.25**2 + 0.6)
# circleFast = plt.Circle(circleFast_center, 0.2, color='r')
# ax1.add_artist(circleRest)
# ax1.add_artist(circleFast)

# Label the circles
# bbox_props = dict(boxstyle="square,pad=0.1", fc="w", ec="k", lw=1)
# arrow_props = dict(facecolor='black', shrink=0.05)
# ax1.annotate("zero energy",
#              xy=circleRest_center,
#              xytext=(-1.6, 1.5),
#              arrowprops=arrow_props,
#              bbox=bbox_props)
# ax1.annotate("any energy",
#              xy=circleFast_center,
#              xytext=(-1.6, 3.0),
#              arrowprops=arrow_props,
#              bbox=bbox_props)

# ---- The quantum states ----
# Draw the first few levels
num_states = 4
E = 1 + 2 * np.arange(0, num_states, 1)
xmax = np.sqrt(E)
xmin = -xmax

for i in range(num_states):
    ax2.hlines(y=E[i], xmin=x_min, xmax=x_max,
               lw=2, linestyle='dashed')

# Draw the first few probability distributions


def wavefunc(n, x):
    ns = np.zeros(n + 1)
    ns[n] = 1
    pre = 1 / np.sqrt(2**n * np.math.factorial(n)) * np.pi**(-0.25)
    wfunc = pre * np.exp(-x**2 / 2) * hermval(x, ns)

    return wfunc


def prob(n, x):
    return wavefunc(n, x)**2

# for i in range(num_states):
#     P = E[i] + 2.5 * prob(i, x)
#     # ax2.plot(x, P, color='r', lw=1.5)
#     ax2.fill_between(x, E[i], P, color='r')


# Output
plt.tight_layout()
plt.savefig("CMvsQM_energy.png")
