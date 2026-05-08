import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AmpKomp_volt = np.array([110e-3, 96e-3, 100e-3, 90e-3, 80e-3, 62e-3, 56e-3, 25e-3, 1e-6])
AmpOkomp_volt = np.array([140e-3, 240e-3, 224e-3, 290e-3, 105e-3, 68e-3, 38e-3, 23e-3, 1e-6])

AmpKomp_volt = np.log10(AmpKomp_volt) * 20
AmpOkomp_volt = np.log10(AmpOkomp_volt) * 20

Hz = np.array([1e3, 5e3, 7e3, 8e3, 10e3, 12e3, 16e3, 25e3, 50e3])
fasKomp_deg = -np.array([1.29, 60, 85, 95, 100, 120, 130, 200, 220]) #vid höda frekvenser estimerades värdena
fasOkomp_deg = -np.array([2, 20, 120, 140, 160, 160, 170, 180, 180])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

ax1.semilogx(Hz, AmpKomp_volt, label='Kompenserad', linestyle='--')
ax1.semilogx(Hz, AmpOkomp_volt, label='Okompenserad', linestyle='--')
ax1.set_title(r'Uppmätt Bodediagram')
ax1.set_ylabel('Amplitud [dB]')
ax1.set_xlabel('Frekvens [Hz]')
ax1.grid(True, which="both", ls="-", alpha=0.5)

ax1.legend()

ax2.semilogx(Hz, fasKomp_deg, label='Kompenserad', linestyle='--')
ax2.semilogx(Hz, fasOkomp_deg, label='Okompenserad', linestyle='--')
ax2.set_title(r'Uppmätt Fas')
ax2.set_ylabel('Fas [deg]')
ax2.set_xlabel('Frekvens [Hz]')
ax2.grid(True, which="both", ls="-", alpha=0.5)

ax2.legend()
plt.show()