import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# function to return coordinates from some basic parameters that describe a circle
def circle_xy(theta, origin_x, origin_y, radius, theta_shift):

    theta = [i + theta_shift for i in theta]

    x = radius * np.cos(theta) + origin_x
    y = radius * np.sin(theta) + origin_y

    return x, y

# make a test plot to get an idea of what the animation will look like
theta = [i for i in np.linspace(0, 2*np.pi, 4)]

xs = []
ys = []

# generate a list of shift values to rotate the polygon
shifts = np.linspace(0,10,100)

for i in shifts:
    x,y = circle_xy(theta, 0, 0, 1, i)
    xs.append(x)
    ys.append(y)

fig, ax = plt.subplots(1,1, figsize=(10,10))

for (x,y) in zip(xs,ys):
    ax.plot(x,y)

fig.savefig('checkplot.png')

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return (line,)

# animation function. This is called sequentially
def animate(j, theta, shifts):
    shift = shifts[j]

    x, y = circle_xy(theta, 0, 0, 1, shift)

    line.set_data(x, y)

    # this line will create the 'trace' effect
    ax.plot(x, y)

    return (line,)

n = 10

theta = np.linspace(0, 2 * np.pi, 4)
shifts = np.linspace(0, 10, 100)

fargs = (theta, shifts)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

line, = ax.plot([], [], lw=1)

ax.axis('off')

ax.set_xlim((-1, 1))
ax.set_ylim((-1, 1))

# call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=200, blit=True, fargs=fargs)
anim.save('rotating_poly_with_trace.gif', writer='imagemagick', fps=10)
# HTML(anim.to_html5_video())