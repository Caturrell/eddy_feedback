import os
import numpy as np
import matplotlib.pyplot as plt

# Define your slopes here
slopes = [0.5, 1.0, 2.0, -1.5]
labels = [f"slope = {m}" for m in slopes]

x = np.linspace(-10, 10, 100)

fig, ax = plt.subplots(figsize=(6, 6))

for m, label in zip(slopes, labels):
    y = m * x
    ax.plot(x, y, label=label)

# Force lines through the origin visually by centering axes there
ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')  # keeps slopes visually accurate/comparable

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.set_title('Comparing slopes through the origin')

plt.tight_layout()

script_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(script_dir, 'compare_slope_lines.png'))

plt.show()