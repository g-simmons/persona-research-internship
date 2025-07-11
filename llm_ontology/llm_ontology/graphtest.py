import matplotlib.pyplot as plt
import numpy as np

x1 = np.linspace(0.0, 5.0, 100)
y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)

fig, ax = plt.subplots(figsize=(5, 3))
fig.subplots_adjust(bottom=0.15, left=0.2)
ax.plot(x1, y1, color='blue')
ax.set_title('Damped oscillation',fontsize=14)
ax.legend(['Damped oscillation curve'])
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
fig.savefig('damped_oscillation.png', dpi=300, bbox_inches='tight')

plt.show()