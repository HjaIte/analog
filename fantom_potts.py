import numpy as np
import matplotlib.pyplot as plt

# 1. Ladda datan från filen
try:
    data = np.loadtxt('Full Amp Biased_sim_cph_feedb.txt', skiprows=1)
except ValueError:
    data = np.genfromtxt('Full Amp Biased_sim_cph_feedb.txt', skip_header=1)

# Extrahera kolumnerna
time = data[:, 0]
ir1 = data[:, 1]
ir12 = data[:, 2]

# 2. Skapa plotten
fig, ax = plt.subplots(figsize=(10, 6))

# Plotta båda linjerna på samma axel (ax)
ax.plot(time, ir1, label='I(RS)', color='tab:blue')
ax.plot(time, ir12, label='I(RL)', color='tab:orange')

# Sätt etiketter och titel
ax.set_xlabel('Tid (s)')
ax.set_ylabel('Ström (A)')
ax.set_title('Fantomnolla på ingång/utgång')

# Lägg till rutnät och teckenförklaring (legend)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# Justera layout och visa/spara
fig.tight_layout()
plt.show()
# plt.savefig('simulation_plot_single_axis.png')