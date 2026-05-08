import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_SIK = pd.read_csv(r'SIK.csv', delimiter=',')
print(df_SIK.head())

df_SK = pd.read_csv(r'SK.CSV', delimiter=',')
print(df_SK.head())

df_SO = pd.read_csv(r'SO.CSV', delimiter=',')

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
