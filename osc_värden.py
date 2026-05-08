import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AmpKomp_volt = np.array([110e-3, 96e-3, 100e-3, 90e-3, 80e-3, 62e-3, 56e-3, 25e-3, 0])
AmpOkomp_volt = np.array([140e-3, 240e-3, 224e-3, 290e-3, 105e-3, 68e-3, 38e-3, 23e-3, 0])
Hz = np.array([1e3, 5e3, 7e3, 8e3, 10e3, 12e3, 16e3, 25e3, 50e-3])

fasKomp_deg = np.array([1.29, 60, 85, 95, 100, 120, 130, 200, 220]) #vid höda frekvenser estimerades värdena
fasOkomp_deg = np.array([2, 20, 120, 140, 160, 160, 170, 180, 180])

df_SIK = pd.read_csv(r'C:\Users\samue\.vscode\analog\ny\analog\SIK.csv', delimiter=',')
print(df_SIK.head())

df_SK = pd.read_csv(r'C:\Users\samue\.vscode\analog\ny\analog\SK.CSV', delimiter=',')
print(df_SK.head())

df_SO = pd.read_csv(r'C:\Users\samue\.vscode\analog\ny\analog\SO.CSV', delimiter=',')

#byt namn

df_SIK.rename(columns={'Source': 'Time (s)', 'CH1': 'Input', 'CH2': 'Output'}, inplace=True)
df_SK.rename(columns={'Source': 'Time (s)', 'CH1': 'Input', 'CH2': 'Kompenserad'}, inplace=True)
df_SO.rename(columns={'Source': 'Time (s)', 'CH1': 'Input', 'CH2': 'Okompenserad'}, inplace=True)


#gör om till ström
df_SIK['Input'] = df_SIK['Input']/10000
df_SIK['Output'] = df_SIK['Output']/100


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

df_SIK.plot(x='Time (s)', y='Input', kind='line', ax=ax1)
df_SIK.plot(x='Time (s)', y='Output', kind='line', ax=ax1)

ax1.set_xlim(-0.001, 0.001)
ax1.set_title('Input och output förstärkare')
ax1.grid(True, linestyle='--')


df_SK.plot(x='Time (s)', y='Kompenserad', kind='line', ax=ax2)
df_SO.plot(x='Time (s)', y='Okompenserad', kind='line', ax=ax2)
ax2.set_title('Stegsvar')
ax2.set_xlim(-0.0005, 0.0005)

ax2.grid(True, linestyle='--') 
plt.show()
