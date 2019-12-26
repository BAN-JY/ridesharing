import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import seaborn as sns


t1 = time.time()
###########################################################################
######### import data from excel file as dataframe #######################################
###########################################################################

list_m_scale_0_01 = []
for m_file in range(0,26):

    list_m_scale_0_01.append(pd.DataFrame(pd.read_csv('./resources/scale_0_01/m{}.csv'.format(m_file))))


list_m_scale_0_05 = []
for i in range(0,26):

    list_m_scale_0_05.append(pd.DataFrame(pd.read_csv('./resources/scale_0_05/m{}.csv'.format(i))))


list_e_scale_0_005 =[]
for e_file in range(0,26):

    list_e_scale_0_005.append(pd.DataFrame(pd.read_excel('./resources/scale_0_005/e{}.xls'.format(e_file, sheet_name='100'))))


############## pair plot when the scale is 5*5, toll is 1 ##################################
total_trip_time_0_01 = list_m_scale_0_01[1].iloc[0, 1:101]
ridesharing_drivers_0_01 = list_m_scale_0_01[1].iloc[1, 1:101]
drive_alone_drivers_0_01 = list_m_scale_0_01[1].iloc[2, 1:101]
drivers_0_01 = list_m_scale_0_01[1].iloc[18,1:101]
passengers_0_01 = list_m_scale_0_01[1].iloc[19,1:101]
travel_time_0_01 = list_m_scale_0_01[1].iloc[20,1:101]

df_0_01 = pd.DataFrame({'total trip time':total_trip_time_0_01,
                        'ridesharing driver': ridesharing_drivers_0_01,
                        'drive-alone drivers': drive_alone_drivers_0_01,
                        'drivers':drivers_0_01,
                        'passengers':passengers_0_01,
                        'travel_time':travel_time_0_01})

sns.set(style="ticks",color_codes=True)
sns.pairplot(data=df_0_01,hue='ridesharing driver')
plt.show()



##################Count time
t2 = time.time()
print(t2-t1)