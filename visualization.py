import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import seaborn as sns


t1 = time.time()
###########################################################################
######### import data from excel file as dataframe #######################################
###########################################################################

################################# scale is 0.01 ###############################################
#import m file as dataframe
list_m = []
for m_file in range(0,26):

    list_m.append(pd.DataFrame(pd.read_csv('./resources/scale_0_01/m{}.csv'.format(m_file))))


# import e file as dataframe
list_e =[]
for e_file in range(0,26):

    list_e.append(pd.DataFrame(pd.read_excel('./resources/scale_0_01/e{}.xls'.format(e_file, sheet_name='100'))))

################################# scale is 0.05 ############################################

list_m_scale_0_05 = []
for i in range(0,26):

    list_m_scale_0_05.append(pd.DataFrame(pd.read_csv('./resources/scale_0_05/m{}.csv'.format(i))))

list_e_scale_0_05 =[]
for e_file in range(0,26):

    list_e_scale_0_05.append(pd.DataFrame(pd.read_excel('./resources/scale_0_05/e{}.xls'.format(e_file, sheet_name='100'))))

################################# scale is 0.1 ###################################################

list_m_scale_0_1 = []
for i in range(0,26):

    list_m_scale_0_1.append(pd.DataFrame(pd.read_csv('./resources/scale_0_1/m{}.csv'.format(i))))


list_e_scale_0_1 =[]
for e_file in range(0,26):

    list_e_scale_0_1.append(pd.DataFrame(pd.read_excel('./resources/scale_0_1/e{}.xls'.format(e_file, sheet_name='100'))))


######################################################################################################
#################### CALCULATION ######################################################################

########################## WITHIN DAY #############################################


time_step = range(1,251)


########################## DAY TO DAY ###############################################

x_day = range(1,101)

#######################(1)(1)(1)(1)(1)(1)(1)(1)(1)(1)(1)(1)(1)###########
#index of rideshare drivers is '1',index id drive-alone drivers is '2'
plt.figure(figsize=(16,9),dpi=120)
##when toll is 0
y0_rideshare_d = list_m[0].iloc[1,1:101]  # index of rideshare drivers is '1'
y0_drivealone_d = list_m[0].iloc[2,1:101]  # index of drive-alone drivers is '2'
#print(y5_rideshare_d)

plt.subplot(1,3,1)
plt.plot(x_day, y0_drivealone_d, color='blue', label='drive-alone')
plt.plot(x_day, y0_rideshare_d, color='red', label='rideshare')

plt.axis((0,100,0,200))
plt.ylabel('Number of rideshare drivers and drive-alone drivers day-to-day',fontsize=16)
plt.title('Toll: 0')


##when toll is 1
y1_rideshare_d = list_m[1].iloc[1,1:101]  # index of rideshare drivers is '1'
y1_drivealone_d = list_m[1].iloc[2,1:101]  # index of drive-alone drivers is '2'
#print(y5_rideshare_d)

plt.subplot(1,3,2)
plt.plot(x_day, y1_drivealone_d, color='blue', label='drive-alone')
plt.plot(x_day, y1_rideshare_d, color='red', label='rideshare')

plt.xlabel('day',fontsize=16)
plt.axis((0,100,0,200))
plt.title('Toll: 1')


##when toll is 25
y25_rideshare_d = list_m[25].iloc[1,1:101]
y25_drivealone_d = list_m[25].iloc[2,1:101]

plt.subplot(1,3,3)
plt.plot(x_day, y25_drivealone_d, color='blue', label='drive-alone')
plt.plot(x_day, y25_rideshare_d, color='red', label='rideshare')

plt.legend(loc='upper right',fontsize=16)
plt.axis((0,100,0,200))
plt.title('Toll: 25')


#plt.savefig('./resources/ad_and_rd.png') #plot
plt.show()

##################///////////////////////////////###########################

###########(2)(2)(2)(2)(2)(2)(2)(2)(2)(2)(2)(2)(2)##########################

##### from excel file, plot number of drivers and passengers ##############################

#index of drivers is '18',index of passenger is '19'
plt.figure(figsize=(16,9),dpi=120)
##when toll is 0
y0_driver = list_m[0].iloc[19,1:101]
y0_passenger = list_m[0].iloc[20,1:101]

plt.subplot(1,3,1)
plt.plot(x_day, y0_driver, color='navy', label='driver')
plt.plot(x_day, y0_passenger, color='brown', label='passenger')

plt.axis((0,100,0,200))
plt.ylabel('Number of drivers and passengers day-to-day',fontsize=16)
plt.title('Toll: 0')

##when toll is 1
y1_driver = list_m[1].iloc[19,1:101]
y1_passenger = list_m[1].iloc[20,1:101]

plt.subplot(1,3,2)
plt.plot(x_day, y1_driver, color='navy', label='driver')
plt.plot(x_day, y1_passenger, color='brown', label='passenger')

plt.axis((0,100,0,200))
plt.xlabel('day',fontsize=16)
plt.title('Toll: 1')


##when toll is 25
y25_driver = list_m[25].iloc[19,1:101]
y25_passenger = list_m[25].iloc[20,1:101]

plt.subplot(1,3,3)
plt.plot(x_day, y25_driver, color='navy', label='driver')
plt.plot(x_day, y25_passenger, color='brown', label='passenger')

plt.axis((0,100,0,200))
plt.legend(loc='upper right',fontsize=16)
plt.title('Toll: 25')

plt.show()

####################///////////////////////////////##########################

width = 0.5

#############(3)(3)(3)(3)(3)(3)(3)(3)(3)(3)(3)(3)########################


# Calculate the detour time as a percentage of trip time

plt.figure(figsize=(16,9),dpi=120)

### when toll is 0 #######################################################
y0_detour_time = list_m[0].iloc[23, 1:101]
y0_trip_time = list_m[0].iloc[0, 1:101]

y0_percentage = []
for i in range(0,100):
    y0_percentage.append(y0_detour_time[i]/y0_trip_time[i])

plt.subplot(1,3,1)
plt.bar(x_day, y0_percentage, width, color='lightsalmon',label='detour time')

plt.ylabel('detour time as a percentage of trip time',fontsize=16)
plt.title('Toll: 0')
plt.axis((0, 100, 0, 0.05))

### when toll is 1 #######################################################
y1_detour_time = list_m[1].iloc[23, 1:101]
y1_trip_time = list_m[1].iloc[0, 1:101]

y1_percentage = []
for i in range(0,100):
    y1_percentage.append(y1_detour_time[i]/y1_trip_time[i])

plt.subplot(1,3,2)
plt.bar(x_day, y1_percentage, width, color='lightsalmon',label='detour time')

plt.xlabel('day',fontsize=16)
plt.title('Toll: 1')
plt.axis((0, 100, 0, 0.05))

### when toll is 25 #######################################################
y25_detour_time = list_m[25].iloc[23, 1:101]
y25_trip_time = list_m[25].iloc[0, 1:101]

y25_percentage = []
for i in range(0,100):
    y25_percentage.append(y25_detour_time[i]/y25_trip_time[i])

plt.subplot(1,3,3)
plt.bar(x_day, y25_percentage, width, color='lightsalmon',label='detour time')

plt.title('Toll: 25')
plt.axis((0, 100, 0, 0.05))




#############################//////////////////////////////###########################

###############(4)(4)(4)(4)(4)(4)(4)(4)(4)(4)(4)(4)(4)(4)#############################

###index of total general cost is '11', index of general cost of expense is '10', general cost of total trip time = total general cost - general cost of expense#######
plt.figure(figsize=(16,9),dpi=120)
### When toll is 0 #################################################################
y0_total_general_cost = list_m[0].iloc[12, 1:101]
y0_cost_expense = list_m[0].iloc[11, 1:101]
y0_cost_total_trip_time = []

for i in range(0,len(y0_total_general_cost)):
    minus = y0_total_general_cost[i] - y0_cost_expense[i]
    y0_cost_total_trip_time.append(minus)

plt.subplot(1,3,1)
plt.bar(x_day, y0_cost_total_trip_time, width, color='yellowgreen', label='general cost of total trip time')
plt.bar(x_day, y0_cost_expense, width, color='indianred', bottom=y0_cost_total_trip_time, label='general cost of expense')

plt.plot(x_day, y0_total_general_cost, color='black', label='total general cost')

plt.ylabel('Total general cost in detail',fontsize=16)
plt.title('Toll: 0')
plt.axis((0, 100, 0, 2000))

####when toll is 1###################################################################
y1_total_general_cost = list_m[1].iloc[12, 1:101]
y1_cost_expense = list_m[1].iloc[11, 1:101]
y1_cost_total_trip_time = []  ### It equals to total general cost minus cost expense

for i in range(0,len(y1_total_general_cost)):
    minus = y1_total_general_cost[i] - y1_cost_expense[i]
    y1_cost_total_trip_time.append(minus)


plt.subplot(1,3,2)
plt.bar(x_day, y1_cost_total_trip_time, width, color='yellowgreen',label='general cost of total trip time')   # in the bottom
plt.bar(x_day, y1_cost_expense, width, color='indianred', bottom=y1_cost_total_trip_time, label='general cost of expense')  # on the top

plt.plot(x_day, y1_total_general_cost, color='black',label='total general cost')

plt.xlabel('day',fontsize=16)
plt.title('Toll: 1')
plt.axis((0,100, 0, 2000))
plt.legend(loc='upper right',fontsize=16)


#### when toll is 25######################################################################
y25_total_general_cost = list_m[25].iloc[12, 1:101]
y25_cost_expense = list_m[25].iloc[11, 1:101]
y25_cost_total_trip_time = []  ### It equals to total general cost minus cost expense

for i in range(0,len(y25_total_general_cost)):
    minus_1 = y25_total_general_cost[i] - y25_cost_expense[i]
    y25_cost_total_trip_time.append(minus_1)

plt.subplot(1,3,3)
plt.bar(x_day, y25_cost_total_trip_time, width, color='yellowgreen',label='general cost of total trip time')   # in the bottom
plt.bar(x_day, y25_cost_expense, width, color='indianred', bottom=y1_cost_total_trip_time, label='general cost of expense')  # on the top

plt.plot(x_day, y25_total_general_cost, color='black',label='total general cost')

plt.xlabel('day')
plt.title('Toll: 25')
plt.axis((0,100, 0, 2000))


plt.show()

#############/////////////////////////////////////////////#################

###################### (5)(5)(5)(5)(5)(5)(5)(5) ########################
### number of drive-alone drivers and rideshare drivers on each routes ####

## index of drive-alone drivers on each route 1, 2, 3 are '23'-'24'-'25' ###
plt.figure(figsize=(16,9),dpi=120)
##  ##  drive-alone drivers route choice  ##  ##
## When toll is 0
y0_n_drivealone_route1 = list_m[0].iloc[25,1:101]
y0_n_drivealone_route2 = list_m[0].iloc[26,1:101]
y0_n_drivealone_route3 = list_m[0].iloc[27,1:101]

plt.subplot(1,3,1)
plt.plot(x_day, y0_n_drivealone_route1, color='lightsalmon', label='route 1')
plt.plot(x_day, y0_n_drivealone_route2, color='orange', label='route 2')
plt.plot(x_day, y0_n_drivealone_route3, color='lightseagreen', label='route 3')

plt.ylabel('Number of drive-alone drivers on each route day-to-day',fontsize=16)
plt.axis((0,100,0,150))
plt.title('Toll: 0')

## When toll is 1
y1_n_drivealone_route1 = list_m[1].iloc[25,1:101]
y1_n_drivealone_route2 = list_m[1].iloc[26,1:101]
y1_n_drivealone_route3 = list_m[1].iloc[27,1:101]

plt.subplot(1,3,2)
plt.plot(x_day, y1_n_drivealone_route1, color='lightsalmon', label='route 1')
plt.plot(x_day, y1_n_drivealone_route2, color='orange', label='route 2')
plt.plot(x_day, y1_n_drivealone_route3, color='lightseagreen', label='route 3')

plt.xlabel('day',fontsize=16)
plt.axis((0,100,0,150))
plt.title('Toll: 1')

## When toll is 25
y25_n_drivealone_route1 = list_m[25].iloc[25,1:101]
y25_n_drivealone_route2 = list_m[25].iloc[26,1:101]
y25_n_drivealone_route3 = list_m[25].iloc[27,1:101]

plt.subplot(1,3,3)
plt.plot(x_day, y25_n_drivealone_route1, color='lightsalmon', label='route 1')
plt.plot(x_day, y25_n_drivealone_route2, color='orange', label='route 2')
plt.plot(x_day, y25_n_drivealone_route3, color='lightseagreen', label='route 3')

plt.axis((0,100,0,150))
plt.legend(loc='upper right',fontsize=16)
plt.title('Toll: 25')


plt.show()

##############(6)(6)(6)(6)(6)############################################

## index of rideshare drivers on each route 1, 2, 3 are '26'-'27'-'28' ##
plt.figure(figsize=(16,9),dpi=120)
# rideshare drivers route choice ##
# When toll is 0
y0_n_rideshare_route1 = list_m[0].iloc[28, 1:101]
y0_n_rideshare_route2 = list_m[0].iloc[29, 1:101]
y0_n_rideshare_route3 = list_m[0].iloc[30, 1:101]

plt.subplot(1,3,1)
plt.plot(x_day, y0_n_rideshare_route1, color='lightsalmon',label='route 1')
plt.plot(x_day, y0_n_rideshare_route2, color='orange', label='route 2')
plt.plot(x_day, y0_n_rideshare_route3, color='lightseagreen', label='route 3')

plt.ylabel('Number of rideshare drivers on each route day-to-day',fontsize=16)
plt.axis((0,100,0,60))
plt.legend(loc='upper right',fontsize=16)
plt.title('Toll: 0')

# When toll is 1
y1_n_rideshare_route1 = list_m[1].iloc[28, 1:101]
y1_n_rideshare_route2 = list_m[1].iloc[29, 1:101]
y1_n_rideshare_route3 = list_m[1].iloc[30, 1:101]

plt.subplot(1,3,2)
plt.plot(x_day, y1_n_rideshare_route1, color='lightsalmon',label='route 1')
plt.plot(x_day, y1_n_rideshare_route2, color='orange', label='route 2')
plt.plot(x_day, y1_n_rideshare_route3, color='lightseagreen', label='route 3')

plt.xlabel('day',fontsize=16)
plt.axis((0,100,0,60))
plt.title('Toll: 1')

# When toll is 25
y25_n_rideshare_route1 = list_m[25].iloc[28, 1:101]
y25_n_rideshare_route2 = list_m[25].iloc[29, 1:101]
y25_n_rideshare_route3 = list_m[25].iloc[30, 1:101]

plt.subplot(1,3,3)
plt.plot(x_day, y25_n_rideshare_route1, color='lightsalmon',label='route 1')
plt.plot(x_day, y25_n_rideshare_route2, color='orange', label='route 2')
plt.plot(x_day, y25_n_rideshare_route3, color='lightseagreen', label='route 3')

plt.axis((0,100,0,60))
plt.title('Toll: 25')

plt.show()


##########/////////////////////////////////#######################################

#####################################################################################
###################### number of rideshare driver simulate with toll from 1 to 25 ###
#####################################################################################

# plot
x_toll = range(0,26)


# import the data of rideshare on the day No.100 with different toll from 0~25
plt.figure(figsize=(15,10),dpi=120)

list_rideshare_0_01 = []  # scale parameter = 0.01
count_0_01 = []
for i in range(0,26):
    for j in range(81,101):
        count_0_01.append(list_m[i].iloc[1,j])
    aver_0_01 = sum(count_0_01)/20
    list_rideshare_0_01.append(aver_0_01)
    count_0_01 = []


x_array = np.array(x_toll)
y_array_0_01 = np.array(list_rideshare_0_01)
z_0_01 = np.polyfit(x_array,y_array_0_01,5)
yvals_0_01 = np.polyval(z_0_01,x_array)


##################################
list_rideshare_0_05 = []  # scale parameter = 0.05
count_0_05 = []
for i in range(0,26):
    for j in range(81,101):
        count_0_05.append(list_m_scale_0_05[i].iloc[1,j])
    aver_0_05 = sum(count_0_05)/20
    list_rideshare_0_05.append(aver_0_05)
    count_0_05 = []


y_array_0_05 = np.array(list_rideshare_0_05)
z_0_05 = np.polyfit(x_array,y_array_0_05,5)
yvals_0_05 = np.polyval(z_0_05,x_array)

####################################
list_rideshare_0_1 = []   # scale parameter = 0.1
count_0_1 = []
for i in range(0,26):
    for j in range(81,101):
        count_0_1.append(list_m_scale_0_1[i].iloc[1,j])
    aver_0_1 = sum(count_0_1)/20
    list_rideshare_0_1.append(aver_0_1)
    count_0_1 = []


y_array_0_1 = np.array(list_rideshare_0_1)
z_0_1 = np.polyfit(x_array,y_array_0_1,5)
yvals_0_1 = np.polyval(z_0_1,x_array)


################################################## fitting results ##

plt.plot(x_toll, yvals_0_01, color = 'red', label = 'scale of 5*5')
plt.plot(x_toll, yvals_0_05, color = 'blue', label = 'scale of 25*25')
plt.plot(x_toll, yvals_0_1, color = 'orange', label = 'scale of 50*50')
# plt.plot(x_toll, yvals_0_005, color = 'lightseagreen', label = 'scale of 2.5*2.5')


plt.scatter(x_toll, list_rideshare_0_01, color = 'red')
plt.scatter(x_toll, list_rideshare_0_05, color = 'blue')
plt.scatter(x_toll, list_rideshare_0_1, color = 'orange')


plt.legend(loc='lower right', fontsize = 16)
plt.ylabel('The average fitting results of toll changes on the number of rideshare vehicles with different scale',fontsize=14)
plt.xlabel('Toll')

plt.axis((0,25,0,70))

plt.show()


# real version
plt.figure(figsize=(15,10),dpi=120)

plt.plot(x_toll, list_rideshare_0_01, color = 'red', label = 'scale of 5*5')
plt.plot(x_toll, list_rideshare_0_05, color = 'blue', label = 'scale of 25*25')
plt.plot(x_toll, list_rideshare_0_1, color = 'orange', label = 'scale of 50*50')
# plt.plot(x_toll, list_rideshare_0_005, color = 'lightseagreen', label = 'scale of 2.5*2.5')

plt.legend(loc='upper right', fontsize = 16)
plt.ylabel('The average results of toll changes on the number of rideshare vehicles with different scale',fontsize=14)
plt.xlabel('Toll')

plt.axis((0,25,0,70))

plt.show()








################################# scale is 0.01 ###############################################
#import m file as dataframe
'''
x_day = range(1,101)


m_scale_0_01_num_60 = pd.DataFrame(pd.read_csv('./resources/m_agent_number60.csv'))

m_scale_0_01_num_90 = pd.DataFrame(pd.read_csv('./resources/m_agent_number90.csv'))

m_scale_0_01_num_120 = pd.DataFrame(pd.read_csv('./resources/m_agent_number120.csv'))

m_scale_0_01_num_150 = pd.DataFrame(pd.read_csv('./resources/m_agent_number150.csv'))

m_scale_0_01_num_180 = pd.DataFrame(pd.read_csv('./resources/scale_0_01/m1.csv'))


############################################### plot #####

plt.figure(figsize=(20,8),dpi=120)

################### agent: 60

plt.subplot(1,5,1)
y1_rideshare_d_60 = m_scale_0_01_num_60.iloc[1,1:101]  # index of rideshare drivers is '1'
y1_drivealone_d_60 = m_scale_0_01_num_60.iloc[2,1:101]  # index of drive-alone drivers is '2'


plt.plot(x_day, y1_drivealone_d_60, color='blue', label='drive-alone')
plt.plot(x_day, y1_rideshare_d_60, color='red', label='rideshare')

plt.ylabel('Number of rideshare drivers and drive-alone drivers day-to-day',fontsize=16)

plt.axis((0,100,0,200))
plt.title('agent: 60')

################### agent: 90

plt.subplot(1,5,2)
y1_rideshare_d_90 = m_scale_0_01_num_90.iloc[1,1:101]  # index of rideshare drivers is '1'
y1_drivealone_d_90 = m_scale_0_01_num_90.iloc[2,1:101]  # index of drive-alone drivers is '2'


plt.plot(x_day, y1_drivealone_d_90, color='blue', label='drive-alone')
plt.plot(x_day, y1_rideshare_d_90, color='red', label='rideshare')

plt.axis((0,100,0,200))
plt.title('agent: 90')

################### agent: 120

plt.subplot(1,5,3)
y1_rideshare_d_120 = m_scale_0_01_num_120.iloc[1,1:101]  # index of rideshare drivers is '1'
y1_drivealone_d_120 = m_scale_0_01_num_120.iloc[2,1:101]  # index of drive-alone drivers is '2'


plt.plot(x_day, y1_drivealone_d_120, color='blue', label='drive-alone')
plt.plot(x_day, y1_rideshare_d_120, color='red', label='rideshare')

plt.xlabel('day',fontsize = 16)
plt.axis((0,100,0,200))
plt.title('agent: 120')

################### agent: 150

plt.subplot(1,5,4)
y1_rideshare_d_150 = m_scale_0_01_num_150.iloc[1,1:101]  # index of rideshare drivers is '1'
y1_drivealone_d_150 = m_scale_0_01_num_150.iloc[2,1:101]  # index of drive-alone drivers is '2'


plt.plot(x_day, y1_drivealone_d_150, color='blue', label='drive-alone')
plt.plot(x_day, y1_rideshare_d_150, color='red', label='rideshare')


plt.axis((0,100,0,200))
plt.title('agent: 150')

################### agent: 180

plt.subplot(1,5,5)
y1_rideshare_d_180 = m_scale_0_01_num_180.iloc[1,1:101]  # index of rideshare drivers is '1'
y1_drivealone_d_180 = m_scale_0_01_num_180.iloc[2,1:101]  # index of drive-alone drivers is '2'


plt.plot(x_day, y1_drivealone_d_180, color='blue', label='drive-alone')
plt.plot(x_day, y1_rideshare_d_180, color='red', label='rideshare')


plt.axis((0,100,0,200))
plt.title('agent: 180')
plt.legend(loc='upper right', fontsize = 16)


plt.show()


########## calculate the ratio of ridesharing drivers #####

ratio_60 = []
for i in range(0,100):
    ratio_60.append(y1_rideshare_d_60[i]/60)

ratio_90 = []
for i in range(0,100):
    ratio_90.append(y1_rideshare_d_90[i]/60)

ratio_120 = []
for i in range(0,100):
    ratio_120.append(y1_rideshare_d_120[i]/60)

ratio_150 = []
for i in range(0,100):
    ratio_150.append(y1_rideshare_d_150[i]/60)

ratio_180 = []
for i in range(0,100):
    ratio_180.append(y1_rideshare_d_180[i]/60)


plt.figure(figsize=(16,10),dpi=120)

new_df = pd.DataFrame({'60': ratio_60,
                       '90': ratio_90,
                       '120': ratio_120,
                       '150': ratio_150,
                       '180': ratio_180})

fig, ax0 = plt.subplots()

pal = sns.cubehelix_palette(5, rot=-.25, light=.7)
sns.set_palette(palette=pal)

ax = new_df.plot(linewidth=0.8)

ax.set_xlabel('Day',fontsize=16)
ax.set_ylabel('The ratio of ridesharing vehicles',fontsize=16)
ax.legend(loc='upper right', fontsize = 12, title="The number of agents")


plt.axis((0,100,0,1))

plt.show()
'''

#### calculation of variance
##################Count time
t2 = time.time()
print(t2-t1)