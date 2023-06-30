import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# column headers
#      Tn	min temperature (°C)
#      Tx	max temperature (°C)
#      Tavg	avg temperature (°C)
#      RH_avg	avg humidity (%)
#      RR	rainfall (mm)
#      ss	duration of sunshine (hour)
#      ff_x	max wind speed (m/s)
#      ddd_x	wind direction at maximum speed (°)
#      ff_avg	avg wind speed (m/s)
#      ddd_car	most wind direction (°)
#      station_id	station id which record the data. Detail of the station can be found in station_detail.csv

df=pd.read_csv('../data/climate_data.csv', encoding='utf-8')

#remove negitive rainfall
df = df[df['RR'] >= 0]

#histagram of rainfall
df['RR'].hist(bins=6, ec='black')
plt.xlabel("Rainfall (mm)")
plt.title("Rainfall")

#metaplot figure closes when hitting enter.
plt.show(block=False)
plt.pause(1)
input()
plt.close()

print(df.describe())

