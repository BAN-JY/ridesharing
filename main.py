
from modsim import *
from modsim import State
import random
import numpy as np
import pandas as pd
import math
from matchingprocess import Matching
import time

##record escape time
t1 = time.time()

# BEGIN ###############################################################################

### Prepare ###

# import the excel file which stores the information of ID, appear time and locations for each agent. These information is fixed during the whole simulation process.
### loop for toll######
excel_file = './resources/generate60.xls'
df_generate = pd.read_excel(excel_file)

new_agentID = df_generate.loc[0:400, 'ID']
X_LOCATION = df_generate.loc[0:400, 'X-Location']
Y_LOCATION = df_generate.loc[0:400, 'Y-Location']


for toll_fee in range(0,26):

    ###############################################################################
    #############make a System object to collect gobal parameters##################
    ###############################################################################
    def make_system():
        #  we need a System object to store the parameters and initial conditions.
        #  Modeling and Simulation in Python那本书里的 P89有提到怎么用
        N = 180  # the number of agents
        T = 200  # time period of agents' appearance
        t0 = 0  # the start time time in a day
        t_end = 300  # the end time step in a day
        day = 100  # how many days will be simulated
        O_x = 500
        O_y = 250
        matching_t = 0  # time-delay for each matching process between a passenger and a driver
        C_g = 0.5  # capacity of general road
        C_h = 1-C_g  # capacity of HOV lane
        beta_s = 0.5  # value of time for rideshare
        beta_d = 0.5  # value of time for drive-alone
        beta_t = 0.5  # Vot for all agents※(not used)
        miu_s = 1  # scale parameter of rideshare
        miu_d = 1  # scale parameter of drive-alone
        miu = 1  # scale parameter of mode choice
        miu_scale = 0.01                                    ### scale is 5*5 ###
        # we do not consider inconveniences of ride-sharing mode, so beta_s =beta_d =beta_t
        beta_r = 0.0  # (not used)
        sigma = 0.0  # external cost:in order to remove unobservable parameter, assume sigma= 0
        toll = toll_fee  # toll policy (value = 0,1,25, three toll level)
        tfg = 6.0  # free flow travel time of general road
        tfh = 6.0  # free flow travel time of HOV lane
        rate_D = 0.7  # 70% of agents are drivers
        v_cap =1  # capacity of vehicle(the number of passengers' seat)
        matchingAlgo = 'fifo'  # change it to different rule of matching 'fifo', 'rank', ... lamp
        init = State(n_p=0, n_rd=0, n_ad=0)  #initial the number of passengers, ridesharing drivers, drive-alone drivers,vehicles who use HOV-lane and vehicles who use General-lane
        ema_n = 3 # finite number of memory capacity for each agent, 当天数小于ema_n时，平均过往经验，当天数大于ema_n时，用ema_n计算
        #ev_p_init = -5.0
        #ev_d_init = -5.0
        #hist= np.random.normal(loc=-5.0, scale=5, size=N)

        return System(init = init, t0 = t0, t_end = t_end, day = day, O_x = O_x, O_y = O_y, matching_t = matching_t, N = N, T = T, C_g = C_g, C_h = C_h, beta_s = beta_s, beta_d = beta_d, beta_t = beta_t,beta_r = beta_r, miu_s = miu_s, miu_d = miu_d, miu = miu, miu_scale = miu_scale, sigma = sigma, toll = toll, tfg = tfg, tfh = tfh, rate_D = rate_D, v_cap = v_cap, matchingAlgo = matchingAlgo,ema_n = ema_n)


    system = make_system()

    ######################################################################################################################
    #calculate travel time of each link and route, route choice utility(share$alone), route choice probility(share&alone), mode choice probility################################################################################################
    ######################################################################################################################

    ############ calculate the distance to origin with inputting the id number of agents #############################

    def distance_to_origin(df, id):

        id_index = (df.loc[df['ID'] == id]).index[0]

        # this function is for calculating the distance between the agents (drivers and passengers) and origin
        agent_x_location = df.loc[id_index, 'X-Location']
        agent_y_location = df.loc[id_index, 'Y-Location']

        distance_to_o = math.sqrt(math.pow(agent_x_location-system.O_x, 2) + math.pow(agent_y_location-system.O_y, 2))

        return distance_to_o

    def find_passenger(df, t):
        distance_min = []
        passenger_ID = []

        for iter_t in range(1, t):
            role = df.loc[iter_t, 'P_D']  # DataFrame.loc[行，列]
            partner = df.loc[iter_t, 'partner']
            pax_id = df.loc[iter_t, 'ID']
            x_location = df.loc[iter_t, 'X-Location']
            y_location = df.loc[iter_t, 'Y-Location']
            D_x_location = df.loc[t, 'X-Location']
            D_y_location = df.loc[t, 'Y-Location']

            if role == 'P' and partner == '-':

                distance_min.append(math.sqrt(math.pow(x_location - D_x_location, 2) + math.pow(y_location - D_y_location, 2)))
                passenger_ID.append(pax_id)
            else:
                pass

        result_min = min(distance_min)
        index_ID = distance_min.index(result_min)
        pas_id = passenger_ID[index_ID]

        return result_min, pas_id

    ############################################################################################################
    ######### calculate the result of travel time on each link and each route ##################################

    def Parameter_calculation(df,t,system):
        # get information of number of vehicles for each lane at present state (time step t)
        # The calculation of number of vehicles base on agent's expectation, not the actual number of vehicles that the agent experienced

        N_o1 = df.loc[t, 'n_o1']  # number of vehicles on link O-1 at time step t
        N_1d = df.loc[t, 'n_1d']  # number of vehicles on link 1-D at time step t
        N_o2 = df.loc[t, 'n_o2']  # number of vehicles on link O-2 at time step t
        N_2d = df.loc[t, 'n_2d']  # number of vehicles on link 2-D at time step t
        N_21 = df.loc[t, 'n_21']  # number of vehicles on link 2-1 at time step t

        # calculate expected travel time based on present state(time step t )
        t_o1 = system.tfg + N_o1 / system.C_h  # link O-1 travel time
        t_1d = system.tfh + N_1d / system.C_h  # link 1-D 上的 travel time
        t_o2 = system.tfg + N_o2 / system.C_g  # link O-2 上的 travel time
        t_2d = system.tfg + N_2d / system.C_g  # link 2-D 上的 travel time
        t_21 = system.tfg + N_21 / system.C_g  # link 2-1 上的 travel time

        # calculate travel time on each route
        t_route1 = t_o1 + t_1d
        t_route2 = t_o2 + t_21 + t_1d
        t_route3 = t_o2 + t_2d

        return (t_o1, t_1d, t_o2, t_2d, t_21,
                t_route1, t_route2, t_route3)

    ####################################################################################################
    ########### calculate the utility and posibility of drive-alone #####################################

    def Parameter_drivealone_data(df,t,system):

        P_C = Parameter_calculation(df,t,system)
        t_route1 = P_C[5]
        t_route2 = P_C[6]
        t_route3 = P_C[7]

        driver_o = distance_to_origin(df, df.loc[t, 'ID'])

        # utility function of drive-alone drivers
        V_1d = - system.beta_d * t_route1 - 3 * system.toll - system.beta_d * system.miu_scale * driver_o  # drive-alone drivers 选择 Route 1 的 utility
        V_2d = - system.beta_d * t_route2 - 2 * system.toll - system.beta_d * system.miu_scale * driver_o  # Route 2
        V_3d = -system.beta_d * t_route3 - system.beta_d * system.miu_scale * driver_o  # ..........................Route 3...........

        # utility function of drive-alone mode
        V_d = (1 / system.miu_d) * math.log(
            (math.exp(system.miu_d * V_1d) + math.exp(system.miu_d * V_2d) + math.exp(system.miu_d * V_3d)), math.e)

        # logit model of drive-alone on each route
        P_1d = (math.exp(V_1d)) / (math.exp(V_1d) + math.exp(V_2d) + math.exp(V_3d))
        P_2d = (math.exp(V_2d)) / (math.exp(V_1d) + math.exp(V_2d) + math.exp(V_3d))
        P_3d = (math.exp(V_3d)) / (math.exp(V_1d) + math.exp(V_2d) + math.exp(V_3d))

        t_DO = system.miu_scale * driver_o

        return (V_1d, V_2d, V_3d,
                V_d,
                P_1d, P_2d, P_3d,
                t_DO)


    ####################################################################################################
    ########### calculate the utility and posibility of rideshare #####################################

    def Parameter_rideshare_data(df,t,system):

        P = Parameter_calculation(df,t,system)

        Find_Passenger = find_passenger(df, t)

        passenger_id = Find_Passenger[1]

        passenger_o = distance_to_origin(df, passenger_id)

        V_1r = - system.beta_s * round(P[5]) - system.beta_s * system.miu_scale * (Find_Passenger[0] + passenger_o)  # ridesharing drivers 选择 Route 1 的 utility
        V_2r = - system.beta_s * round(P[6]) - system.beta_s * system.miu_scale * (Find_Passenger[0] + passenger_o)  # ridesharing drivers 选择 Route 2 的 utility
        V_3r = - system.beta_s * round(P[7]) - system.beta_s * system.miu_scale * (Find_Passenger[0] + passenger_o)  # ridesharing drivers 选择 Route 3 的 utility

        # logit model of rideshare on each route
        P_1r = (math.exp(V_1r)) / (math.exp(V_1r) + math.exp(V_2r) + math.exp(V_3r))
        P_2r = (math.exp(V_2r)) / (math.exp(V_1r) + math.exp(V_2r) + math.exp(V_3r))
        P_3r = (math.exp(V_3r)) / (math.exp(V_1r) + math.exp(V_2r) + math.exp(V_3r))

        V_r = (1 / system.miu_s) * math.log((math.exp(system.miu_s * V_1r) + math.exp(system.miu_s * V_2r) + math.exp(system.miu_s * V_3r)), math.e)

        t_DP = system.miu_scale * Find_Passenger[0]
        t_PO = system.miu_scale * passenger_o

        return (V_1r, V_2r, V_3r,
                P_1r, P_2r, P_3r,
                V_r,
                t_DP, t_PO)

    def calculate_drive_mode_choice(df,t,system):
        R_P = Parameter_rideshare_data(df,t,system)
        D_P = Parameter_drivealone_data(df,t,system)

        # return the result of V_r and V_d
        V_r = R_P[6]
        V_d = D_P[3]

        # logit model of choosing rideshare and drive-alone
        P_r = (math.exp(system.miu * V_r)) / (math.exp(system.miu * V_r) + math.exp(system.miu * V_d))
        P_d = (math.exp(system.miu * V_d)) / (math.exp(system.miu * V_r) + math.exp(system.miu * V_d))

        return P_r, P_d

    def Return_parameter_result(df,t,system):
        # Caculate Parameter function here in order to reduce caculation time

        Parameter_Caculation = Parameter_calculation(df,t,system)

        # Parameter_Caculation[] here is a local variable

        df.loc[t, 'travel time_link O-1'] = round(Parameter_Caculation[0])
        df.loc[t, 'travel time_link 1-D'] = round(Parameter_Caculation[1])
        df.loc[t, 'travel time_link O-2'] = round(Parameter_Caculation[2])
        df.loc[t, 'travel time_link 2-D'] = round(Parameter_Caculation[3])
        df.loc[t, 'travel time_link 2-1'] = round(Parameter_Caculation[4])
        df.loc[t, 'travel time_route 1'] = round(Parameter_Caculation[5])
        df.loc[t, 'travel time_route 2'] = round(Parameter_Caculation[6])
        df.loc[t, 'travel time_route 3'] = round(Parameter_Caculation[7])

    def Return_drivealone_result(df,t,system):

        P_D = Parameter_drivealone_data(df,t,system)

        df.loc[t, 'utility_drive-alone_route 1'] = P_D[0]
        df.loc[t, 'utility_drive-alone_route 2'] = P_D[1]
        df.loc[t, 'utility_drive-alone_route 3'] = P_D[2]
        df.loc[t, 'utility_drive-alone'] = P_D[3]
        df.loc[t, 'P_drive-alone_route 1'] = P_D[4]
        df.loc[t, 'P_drive-alone_route 2'] = P_D[5]
        df.loc[t, 'P_drive-alone_route 3'] = P_D[6]

    def Return_rideshare_result(df,t,system):

        P_R = Parameter_rideshare_data(df,t,system)

        df.loc[t, 'utility_ridesharing_route 1'] = P_R[0]
        df.loc[t, 'utility_ridesharing_route 2'] = P_R[1]
        df.loc[t, 'utility_ridesharing_route 3'] = P_R[2]
        df.loc[t, 'P_ridesharing_route 1'] = P_R[3]
        df.loc[t, 'P_ridesharing_route 2'] = P_R[4]
        df.loc[t, 'P_ridesharing_route 3'] = P_R[5]
        df.loc[t, 'utility_rideshare'] = P_R[6]


    #############################################################################
    #####################ridesharing driver route choice#########################
    #############################################################################

    def RD_route_choice(df,t,system):
        " " " Calculate the result of ridesharing route choosing from probability P_1r[16], P_2r[17], P_3r[18]" " "
        # Generate a random number
        random_number1 = random.random()
        df.loc[t,'Random_RD_route_choice'] = random_number1
        # Caculate Parameter function here in order to reduce caculation time
        R_P = Parameter_rideshare_data(df,t,system)

        link_t = Parameter_calculation(df,t,system)
        # Parameter_Caculation[] here is a local variable

        route1 = R_P[3]
        route2 = R_P[4]
        t_o1 = round(link_t[0])
        t_1d = round(link_t[1])
        t_o2 = round(link_t[2])
        t_2d = round(link_t[3])
        t_21 = round(link_t[4])

        t_DP = R_P[7]
        t_PO = R_P[8]

        if (random_number1 > 0) & (random_number1 < route1):
            return 'Route1', t_o1, t_1d, 0, t_DP, t_PO
        elif (random_number1 > route1) & (random_number1<(route1+route2)):
            return 'Route2', t_o2, t_21, t_1d, t_DP, t_PO
        else:
            return 'Route3', t_o2, t_2d, 0, t_DP, t_PO


    ##############################################################################
    #####################drive-alone driver route choice##########################
    ##############################################################################

    def DA_route_choice(df, t, system):
        " " " Calculate the result of drive-alone route choosing from probability P_1d[19],P_2d[20],P_3d[21]" " "
        random_number2 = random.random()
        df.loc[t,'Random_DA_route_choice'] = random_number2
        # Caculate Parameter function here in order to reduce caculation time
        P_D = Parameter_drivealone_data(df,t,system)

        link_t = Parameter_calculation(df,t,system)


        route1 = P_D[4]
        route2 = P_D[5]
        t_o1 = round(link_t[0])
        t_1d = round(link_t[1])
        t_o2 = round(link_t[2])
        t_2d = round(link_t[3])
        t_21 = round(link_t[4])

        t_DO = P_D[7]

        # Parameter_Caculation[] here is a local variable
        if (random_number2 > 0) & (random_number2 < route1):
            return 'Route1', t_o1, t_1d, 0, t_DO
        elif (random_number2 > route1) & (random_number2<(route1+route2)):
            return 'Route2', t_o2, t_21, t_1d, t_DO
        else:
            return 'Route3', t_o2, t_2d, 0, t_DO


    #############################################################################
    ###############driver mode choice ###########################################
    #############################################################################
    def D_mode_choice(df, t, system):
        " " " Calculate the result of driver mode choice from probability P_r[22], P_d[23] " " "
        random_number3 = random.random()
        df.loc[t,'Random_mode choice'] = random_number3
        # Caculate Parameter function here in order to reduce caculation time
        mode_c = calculate_drive_mode_choice(df,t,system)

        df.loc[t,'P_rideshare'] = mode_c[0]
        df.loc[t,'P_drive-alone'] = mode_c[1]

        P_r = mode_c[0]
        # Parameter_Caculation[] here is a local variable
        if (random_number3 > 0) & (random_number3 < P_r):
            return 'ride-sharing'
        else:
            return 'drive-alone'


    ##########################################################################################################
    ########Calculate the result of each parameter with different route choice when match before delay #######
    ##########################################################################################################

    def RD_parameter_calculate(t, df, system):
        # pre-calculate agent's choice
        RD_Route_Choice = RD_route_choice(df, t, system)  # rideshare 的route choice

        t_DP = round(RD_Route_Choice[4])
        t_PO = round(RD_Route_Choice[5])

        Distance_O = distance_to_origin(df, df.loc[t, 'ID'])
        t_DO = round(system.miu_scale * Distance_O)


        if t < system.matching_t:
            pass  # pass ... do nothing
        else:
            # add rs vehicle into the transport system based on the decision in system.matching_t ago
            if df.loc[t, 'mode'] == 1:

                df.loc[t, 'rd_v'] = 1
                df.loc[t, 'detour time'] = t_DP + t_PO - t_DO
                df.loc[t, 'pick up time'] = t_DP + t_PO
                # 在'rd_v'(在rideshare vehicle 列上记上1
                # time_get_out = round(t + RD_Route_Choice[1])

                # record the time when the vehicle get out of some nodes:
                if RD_Route_Choice[0] == 'Route1':
                    time_get_out_o1 = round(t + t_DP + t_PO + RD_Route_Choice[1])
                    time_get_out_1d = round(time_get_out_o1 + RD_Route_Choice[2])  # 离开link1-d的时间

                    df.loc[t, 'route'] = 'Route1'

                    # record the number of vehicles on the links
                    # 仿真过程 START------------>

                    df.loc[t + t_DP + t_PO + 1:time_get_out_o1, 'n_o1'] += 1  # 在该辆车所出现的时间的车辆数都加1
                    df.loc[time_get_out_o1, 'get_out_o1'] += 1

                    df.loc[time_get_out_o1 + 1:time_get_out_1d, 'n_1d'] += 1
                    df.loc[time_get_out_1d, 'get_out_1d'] += 1

                    df.loc[t + t_DP + t_PO + 1:time_get_out_1d, 'n_rideshare'] += 1  # Compute the number of vehicles in network by time step
                    # 仿真过程END<---------------

                    df.loc[t, 'travel'] = round(RD_Route_Choice[1] + RD_Route_Choice[2])
                    # Total trip time of  matched driver = matching time + travel time on route
                    df.loc[t, 'trip_time'] = df.loc[t, 'travel'] + t_DP + t_PO

                elif RD_Route_Choice[0] == 'Route2':

                    df.loc[t, 'route'] = 'Route2'
                    time_get_out_o2 = round(t + t_DP + t_PO + RD_Route_Choice[1])
                    time_get_out_21 = round(time_get_out_o2 + RD_Route_Choice[2])
                    time_get_out_1d = round(time_get_out_21 + RD_Route_Choice[3])

                    # 仿真过程START------------>
                    df.loc[t + t_DP + t_PO + 1:time_get_out_o2, 'n_o2'] += 1
                    df.loc[time_get_out_o2, 'get_out_o2'] += 1

                    df.loc[time_get_out_o2 + 1:time_get_out_21, 'n_21'] += 1
                    df.loc[time_get_out_21, 'get_out_21'] += 1

                    df.loc[time_get_out_21 + 1:time_get_out_1d, 'n_1d'] += 1  # 记录选择route2后对link 1-d产生的车
                    df.loc[time_get_out_1d, 'get_out_1d'] += 1

                    df.loc[t + t_DP + t_PO + 1:time_get_out_1d,'n_rideshare'] += 1  # Compute the number of vehicles in network by time step
                    # 仿真过程END<---------------

                    df.loc[t, 'travel'] = round(RD_Route_Choice[1] + RD_Route_Choice[2] + RD_Route_Choice[3])
                    df.loc[t, 'trip_time'] = df.loc[t, 'travel'] + t_DP + t_PO

                elif RD_Route_Choice[0] == 'Route3':
                    # 台数更新
                    df.loc[t, 'route'] = 'Route3'
                    # record the number of vehicles on the links）

                    time_get_out_o2 = round(t + t_DP + t_PO + RD_Route_Choice[1])
                    time_get_out_2d = round(time_get_out_o2 + RD_Route_Choice[2])

                    # 仿真过程START--------------->
                    df.loc[t + t_DP + t_PO + 1:time_get_out_o2, 'n_o2'] += 1
                    df.loc[time_get_out_o2, 'get_out_o2'] += 1

                    df.loc[time_get_out_o2 + 1:time_get_out_2d, 'n_2d'] += 1
                    df.loc[time_get_out_2d, 'get_out_2d'] += 1

                    df.loc[t + t_DP + t_PO + 1:time_get_out_2d,'n_rideshare'] += 1  # Compute the number of vehicles in network by time step
                    # 仿真过程END<------------------

                    df.loc[t, 'travel'] = round(RD_Route_Choice[1] + RD_Route_Choice[2])
                    # Total trip time of  matched driver = matching time + travel time on route
                    df.loc[t, 'trip_time'] = df.loc[t, 'travel'] + t_DP + t_PO

                # store the matched passenger travel time (same with the matched driver)
                # 被match的passenger的旅行时间一栏等于match的driver的旅行时间一栏

            else:
                pass  # matching time 之前没有match的话，则在这一步不需要计算

        return


    ##################################################################################################################
    ########Calculate the result of each parameter with different route choice while drive mode is drive-alone #######
    ##################################################################################################################


    def DA_parameter_calculate(t, df, system):
        # pre-calculate agent's choice
        DA_Route_Choice = DA_route_choice(df, t, system)  # drive lone the route choice
        t_DO = round(DA_Route_Choice[4])

        if DA_Route_Choice[0] == 'Route1':

            df.loc[t, 'route'] = 'Route1'

            time_get_out_o1 = round(t + t_DO + DA_Route_Choice[1])
            time_get_out_1d = round(time_get_out_o1 + DA_Route_Choice[2])

            # simulation process START---------------->
            df.loc[t + t_DO + 1:time_get_out_o1, 'n_o1'] += 1
            df.loc[time_get_out_o1, 'get_out_o1'] += 1

            df.loc[time_get_out_o1 + 1:time_get_out_1d, 'n_1d'] += 1
            df.loc[time_get_out_1d, 'get_out_1d'] += 1

            df.loc[t + t_DO + 1:time_get_out_1d, 'n_drivealone'] += 1
            # simulation process END<-----------------

            df.loc[t, 'travel'] = round(DA_Route_Choice[1] + DA_Route_Choice[2])
            df.loc[t, 'trip_time'] = df.loc[t, 'travel'] + t_DO# drive-alone driver does not need to wait for partners
            df.loc[t, 'toll'] = 3 * system.toll

        elif DA_Route_Choice[0] == 'Route2':

            df.loc[t, 'route'] = 'Route2'

            time_get_out_o2 = round(t + t_DO + DA_Route_Choice[1])
            time_get_out_21 = round(time_get_out_o2 + DA_Route_Choice[2])
            time_get_out_1d = round(time_get_out_21 + DA_Route_Choice[3])

            # simulation process START------------------------->
            df.loc[t + t_DO + 1:time_get_out_o2, 'n_o2'] += 1
            df.loc[time_get_out_o2, 'get_out_o2'] += 1

            df.loc[time_get_out_o2 + 1:time_get_out_21, 'n_21'] += 1
            df.loc[time_get_out_21, 'get_out_21'] += 1

            df.loc[time_get_out_21 + 1:time_get_out_1d, 'n_1d'] += 1
            df.loc[time_get_out_1d, 'get_out_1d'] += 1

            df.loc[t + t_DO + 1:time_get_out_1d, 'n_drivealone'] += 1
            # simulation process END<---------------------------

            df.loc[t, 'travel'] = round(DA_Route_Choice[1] + DA_Route_Choice[2] + DA_Route_Choice[3])
            df.loc[t, 'trip_time'] = df.loc[t, 'travel'] + t_DO
            df.loc[t, 'toll'] = 2 * system.toll

        elif DA_Route_Choice[0] == 'Route3':

            df.loc[t, 'route'] = 'Route3'
            # record the number of vehicles on the links

            time_get_out_o2 = round(t + t_DO + DA_Route_Choice[1])
            time_get_out_2d = round(time_get_out_o2 + DA_Route_Choice[2])  # 离开link1-d的时间

            # simulation process START--------------------->
            df.loc[t + t_DO + 1:time_get_out_o2, 'n_o2'] += 1
            df.loc[time_get_out_o2, 'get_out_o2'] += 1

            df.loc[time_get_out_o2 + 1:time_get_out_2d, 'n_2d'] += 1
            df.loc[time_get_out_2d, 'get_out_2d'] += 1

            df.loc[t + t_DO + 1:time_get_out_2d, 'n_drivealone'] += 1
            # simulation process END<------------------------

            df.loc[t, 'travel'] = round(DA_Route_Choice[1] + DA_Route_Choice[2])
            # Total trip time of  matched driver = matching time + travel time on route
            df.loc[t, 'trip_time'] = df.loc[t, 'travel'] + t_DO

        return


    ###########################################################################
    ############ within-day time loop update function##########################
    ###########################################################################


    def update_func(state, t, system, df):
        # pre-calculate agent's choice

        N_p, N_rd, N_ad = state

        # use for to update the record of N_p (number of passenger on the network), N_rd (number of rideshare drivers on the network) and N_ad (number of drive-alone drivers)

        # for t で回した時に（t-1）でstateに保存した結果をｔの時に利用される
        # state の中の変数の値(run_simulation中で定義しているframe.locの値)を順番に左側の変数に与える
        #   N_p, N_rd, N_ad, N_o1, N_1d, N_o2, N_2d, N_21 are local variables, different from state variable

        # store parameter at each time step

        Return_parameter_result(df, t, system)

        #################3パターン#################################################################
        '''
        In each of the three patterns, the determination of whether or not a ridesharing vehicle due to the end of the matching process appears at this time step,
        It is necessary to judge whether the car that appeared before leaves the lane
        '''
        #

        ############### ①没有agents出现的場合# P_D=0)###############################################
        if df.loc[t, 'P_D'] == '-':
            N_p += 0
            N_rd += 0
            N_ad += 0

            # df is dataframe, 'P-D'列t行的数为'-'时，即没有agents出现的场合
            # N_rv = frame.iloc[(t- system.matching_t) , n_rd]

            # If there is no agent appear in this time step, all parameters are recorded nothing
            df.loc[t, 'choice?'] = '-' # choice = 1 (agent choose to be driver)
            df.loc[t, 'mode'] = '-' # mode = 0 (there is no passenger waiting, driver have to drive-alone), mode = 1 (there are passengers waiting, driver choose ridesharing), mode = x (there are passengers waiting, driver choose drive-alone)

            # Determine if ridesharing vehicle comes out of matching
            # Record the time of disappearance from ②lane when spawning ② Update the increase of vehicles on the road

            # If a ridesharing vehicle appears:
            # if not yet time for matching

        elif df.loc[t, 'P_D'] == 'P':

            N_p += 1
            N_rd += 0
            N_ad += 0

            df.loc[t, 'choice?'] = '-'
            df.loc[t, 'mode'] = '-'


        elif df.loc[t, 'P_D'] == 'D':

            Return_drivealone_result(df,t,system)

            N_p += 0

            df.loc[t, 'choice?'] = 1
         ###########如果有乘客正在等待着dr
            if N_p != 0:

                Return_rideshare_result(df,t,system)

                # calculate the driver mode choice
                D_Mode_Choice = D_mode_choice(df, t, system)

                # ----- matching process start from here, some of the function within it from matchingprocess
                if D_Mode_Choice == 'ride-sharing':

                    df.loc[t, 'mode'] = 1

                    # calculate the result according to different route choice. This process must be after df.loc[t, 'mode'] = 1
                    RD_parameter_calculate(t, df, system)

                    # FIRST: run the matching process, get partner

                    mat = Matching(t, df, N_rd, N_p, system.v_cap, system.N, system.O_x, system.O_y)
                    mat.run_matching()
                    N_rd = mat.get_N_rd()
                    N_p = mat.get_N_p()
                    df.loc[t, 'mode'] = mat.get_mode()

                    # store matching result
                    if len(mat.get_partners()) == 0:
                        df.loc[t,'partner'] = 'No_partner'
                    elif len(mat.get_partners()) == 1:
                        df.loc[t,'partner'] = mat.get_partners()
                    else:
                        df.loc[t,'partner'] = 'Too_many_partner'

                    # calculate the time from D to P
                    distance_DP = mat.min_distance()
                    t_DP = system.miu_scale * distance_DP

                    # record matched driver to passenger
                    for partner in mat.get_partners():

                        #ma.get_partners returns partners
                        df.loc[df['ID'] == partner, 'partner'] = df.loc[t, 'ID']
                        #Add 'ID' to column for 'parter'
                        # store matching time right after matching process is done Lamp
                        # If there is time delay, maybe matching time should be calculated and stored later ... lamp
                        #Add matchtime to ID line when partner is ID、
                        df.loc[df['ID'] == partner, 'WaitingTime'] = t - (df.loc[df['ID'] == partner]).index[0] + round(t_DP)

                        # the travel time of being picked up passenger
                        df.loc[(df.ID == (df.loc[t, 'partner'])), 'travel'] = df.loc[t, 'travel']
                        df.loc[(df.ID == (df.loc[t, 'partner'])), 'trip_time'] = df.loc[t, 'trip_time'] + df.loc[
                            (df.ID == ((df.loc[t, 'partner']))), 'WaitingTime'] - round(t_DP)

                    # ----- matching process end here


            # At this time, the generated driver will not do route choice because of matching time on matching process
                elif D_Mode_Choice == 'drive-alone':

                    # the route choice model is used in here: DA_parameter_calculate

                    N_ad += 1

                    # record the route choice of drive-alone driver
                    df.loc[t, 'mode'] = 0

                    DA_parameter_calculate(t, df, system)

            else:
                # if there is no passenger waiting for matching at that time, then driver will definitely choose drive-alone

                df.loc[t, 'utility_ridesharing_route 1'] = 0
                df.loc[t, 'utility_ridesharing_route 2'] = 0
                df.loc[t, 'utility_ridesharing_route 3'] = 0
                df.loc[t, 'P_ridesharing_route 1'] = 0
                df.loc[t, 'P_ridesharing_route 2'] = 0
                df.loc[t, 'P_ridesharing_route 3'] = 0
                df.loc[t, 'utility_rideshare'] = 0

                N_ad += 1
                df.loc[t, 'mode'] = 'x'  # x means driver who choose drive-alone because no passenger accured

                DA_parameter_calculate(t,df,system)


        return State(n_p=N_p, n_rd=N_rd, n_ad=N_ad)


    #########################################################################################################################
    # we can call update_fuc at one time step based on initial situation( update_state = update_func(init,0,system))

    #####################################################################################
    #########simulate the model over a sequence of time steps############################
    #####################################################################################
    # RUN simulation by each time step: store the result of N_p, N_rd, N_ad

    def run_simulation(update_func, df):
        choushibao = make_system()
        c = choushibao.init.index
        frame = DataFrame(columns = c)
         # set colums of dataframe to index of initial state
        frame.loc[choushibao.t0] = choushibao.init  # data初始值的设定＝init

        # run simulation by each time step ... lamp
        for t in range(choushibao.t0, choushibao.t_end-1):  # the number of time step is system.t_end (0～t_end-1)
            frame.loc[t+1] = update_func(frame.loc[t], t, choushibao, df)  ##update_function的第0項的結果
            print('time step:{}'.format(t))

        return frame



    def store_result(df):
        run_sim = run_simulation(update_func,df)
        result = pd.concat([df, run_sim], axis=1)  # connect the n_rd, n_p and n_ad with df
        return result

    #################################################################################
    ######################### metric的更新 #########################################
    #################################################################################

    def metrics_update_func(system, result):
        # 各种指标的计算，即 m 表

        ## 各种评价指标的计算
        total_trip_time = (result['trip_time']).sum()  #  计算当天所有trip time的总和，trip time包括matching time, waiting time of passenger和travel time
        travel_time = (result['travel']).sum()  #  当天所有travel time之和汇总到m表中(第1天~第30天)
        waiting_time = (result['WaitingTime']).sum()  #  当天所有乘客waiting time之和

        detour_time = (result['detour time']).sum()

        average_pick_up_time = (result['pick up time']).sum() / (result['mode'] ==1).sum()

        if len(average_pick_up_time) == 0:
            average_pick_up_time = 0
        else:
            pass



        # drive alone drivers 选择 route 1的数，为了求toll
        n_drive_alone_1 = (((result['mode'] =='x')|(result['mode'] == 0)) & (result['route'] == 'Route1')).sum()
        # drive alone drivers 选择 route 2的数, 为了求toll
        n_drive_alone_2 = (((result['mode'] =='x')|(result['mode'] == 0)) & (result['route'] == 'Route2')).sum()
        # drive alone drivers 选择 route 3的数, 为了求toll
        n_drive_alone_3 = (((result['mode'] == 'x') | (result['mode'] == 0)) & (result['route'] == 'Route3')).sum()
        # 要付toll 的车辆数
        tolled_vehicle = n_drive_alone_1 + n_drive_alone_2

        # ridesharing drivers 选择 route 1 的数量
        n_ridesharing_1 = ((result['mode'] ==1) & (result['route'] == 'Route1')).sum()
        # ridesharing drivers 选择 route 2 的数量
        n_ridesharing_2 = ((result['mode'] ==1) & (result['route'] == 'Route2')).sum()
        # ridesharing drivers 选择 route 3 的数量
        n_ridesharing_3 = ((result['mode'] ==1) & (result['route'] == 'Route3')).sum()

        total_revenue = (result['toll']).sum()  # revenue只包含toll
        total_general_cost = total_trip_time * system.beta_d + total_revenue

        n_ridesharing = (result['n_rd']).max()  # n_rd表示此时此刻之前有过的rideshare数，因此当天ridesharing 数应当取'n_rd'中最大的
        #total_system_matching_time = n_ridesharing * (system.v_cap+1) * system.matching_t #rideshare司机和乘客match time之和
        n_drive_alone = (result['n_ad']).max() #与n_ridesharing的计算同理
        n_not_matched_passenger = ((result['P_D'] == 'P') & (result['partner'] == '-')).sum()

        rideshare_ratio = n_ridesharing/system.N

        ### number of vehicles on each link###
        n_o1 = (result['route'] == 'Route1').sum()
        n_1d = ((result['route'] == 'Route1')|(result['route'] == 'Route2')).sum()
        n_o2 = ((result['route'] == 'Route3')|(result['route'] == 'Route2')).sum()
        n_21 = (result['route'] == 'Route2').sum()
        n_2d = (result['route'] == 'Route3').sum()


        # Lamp revised functions of avg travel time on each lane
        # b/c Lamp store travel time for passenger as well
        # 寻找dataframe中走route1,route2,或route3的drivers，result[bool]，isin是pandas的函数
        veh_on_1 = result[result.route.isin(['Route1']) & result.P_D.isin(['D'])]
        veh_on_2 = result[result.route.isin(['Route2']) & result.P_D.isin(['D'])]
        veh_on_3 = result[result.route.isin(['Route3']) & result.P_D.isin(['D'])]

        # vehicle on Route1
        if len(veh_on_1.index) == 0:
            average_travel_time_on_1_d = 0
            variance_travel_time_on_1_d = 0  # len(veh_on_1.index)求vehicle on route1的车辆数
        else:
            average_travel_time_on_1_d = veh_on_1['travel'].sum() / len(veh_on_1.index)
            variance_travel_time_on_1_d = np.var(veh_on_1['travel'])

        # vehicle on Route2
        if len(veh_on_2.index) == 0:
            average_travel_time_on_2_d=0
            variance_travel_time_on_2_d = 0
        else:
            average_travel_time_on_2_d = veh_on_2['travel'].sum() / len(veh_on_2.index)
            variance_travel_time_on_2_d = np.var(veh_on_2['travel'])

        # vehicle on Route3
        if len(veh_on_3.index) == 0:
            average_travel_time_on_3_d=0
            variance_travel_time_on_3_d = 0
        else:
            average_travel_time_on_3_d = veh_on_3['travel'].sum() / len(veh_on_3.index)
            variance_travel_time_on_3_d = np.var(veh_on_3['travel'])

        driver = (result['P_D'] == 'D').sum()
        passenger = (result['P_D'] =='P').sum()

        return (total_trip_time,
                n_ridesharing,
                n_drive_alone,
                rideshare_ratio,
                n_not_matched_passenger,
                n_o1,
                n_1d,
                n_o2,
                n_21,
                n_2d,
                tolled_vehicle,
                total_revenue,
                total_general_cost,
                average_travel_time_on_1_d,
                variance_travel_time_on_1_d,
                average_travel_time_on_2_d,
                variance_travel_time_on_2_d,
                average_travel_time_on_3_d,
                variance_travel_time_on_3_d,
                driver,
                passenger,
                travel_time,
                waiting_time,
                detour_time,
                average_pick_up_time,
                n_drive_alone_1,
                n_drive_alone_2,
                n_drive_alone_3,
                n_ridesharing_1,
                n_ridesharing_2,
                n_ridesharing_3)

    ###############################################################################################
    ###################################### core code###########################################
    ###############################################################################################

    ## preparing dataframe --> not neccessary
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    # set a seed of random function to 0, so it can regenerate ... lamp
    np.random.seed(0)


    # system = make_system()  # 代码一开始定义的各种常数

    ################ MAKE A NEW DF(dataframe) FOR DAY>=1#####################


    ##################make a dictionary to store actual utility for agent i on each day########
    #Agent 𝑖’s actual utility for being a passenger in day 𝑘 ...ev_P
    #Agent 𝑖’s actual utility for being a driver in day 𝑘 ...ev_D

    def update_P_D(k,system,dic_au,hist_p,hist_d):
        ############### on each day, there will be a whole new_df generating (because of the loop is for all agents on that day)

        # call the function to get a new ID list

        # store the result in a new dataframe new_df(index= 'time','new_ID','new_P_D'),
        new_df = pd.DataFrame({'day': k, 'ID': new_agentID, 'P_D': '-', 'X-Location': X_LOCATION, 'Y-Location': Y_LOCATION})

        # loop for all agent in day K to calculate actual/experienced travel travel
        # make a empty dictionary to store ID loop (key: # of agent, value: dataframe au_P_D )

        for i in range(system.N): # number of ID (0～N-1)

            ##############store actual utility of P and D for agent i###############
            if k == 1:

                #on the first day, the initial predicted travel cost will be given
                # the assumed normal distribution is for consider the basic behavior of model
                ev_p_init = random.choice(hist_p)
                ev_d_init = random.choice(hist_d)

                dic_au[i].loc[k,'au_p'] = ev_p_init #actual utility of passenger
                dic_au[i].loc[k,'au_d'] = ev_d_init #.....of driver

                dic_au[i].loc[k, 'ev_p'] = ev_p_init #EMA 得到的utility of passenger
                dic_au[i].loc[k, 'ev_d'] = ev_d_init #.....of driver

            else :

                #print(list((dic[k - 1])[dic[k - 1]['ID'] == i]['P_D'])[0])
                if list((dic[k-1])[dic[k-1]['ID'] == i]['P_D'])[0] == 'P':
                    #如果他之前一天的选择(因为把今天的预测选择储存在了前一天的表格里面)是P，那么他今天的ac_p的计算就是用的前一天情报（mistake,）（这里储存的实际旅行时间#  ）
                    dic_au[i].loc[k, 'au_p'] = -system.beta_t * (list((dic[k-1])[dic[k-1]['ID']==i]['trip_time'])[0])
                    #print((list((dic[k-1])[dic[k-1]['ID']==i]['trip_time']))[0])
                    # actual utility of mode D is given from previous day
                    dic_au[i].loc[k, 'au_d'] = dic_au[i].loc[k-1,'au_d']
                elif list((dic[k-1])[dic[k-1]['ID'] == i]['P_D'])[0] == 'D':
                    dic_au[i].loc[k, 'au_d'] = -system.beta_t * (list(dic[k-1][dic[k-1]['ID']==i]['trip_time'])[0]) - (list(dic[k - 1][dic[k - 1]['ID'] == i]['toll'])[0])

                    # actual utility of mode P is given from previous day
                    dic_au[i].loc[k, 'au_p'] = dic_au[i].loc[k-1, 'au_p']

            ###########store predicted utility for agents' mode choice(to be a P or D) by using ema#######
            # when day k <  n (指数平滑移動平均)
                if k < system.ema_n:
                    dic_au[i].loc[k, 'ev_p'] = (sum(dic_au[i]['au_p'][1:k + 1])) / k
                    dic_au[i].loc[k, 'ev_d'] = (sum(dic_au[i]['au_d'][1:k + 1])) / k

                else: # ema function
                    mu = 2 / float(1 + system.ema_n)
                    # get n sma first the initial EMA value
                    if k == system.ema_n:
                        dic_au[i].loc[k, 'ev_p'] = (sum(dic_au[i]['au_p'][1:k+1]))/system.ema_n #k+1 means k
                        dic_au[i].loc[k, 'ev_d'] = (sum(dic_au[i]['au_d'][1:k+1]))/system.ema_n
                    # after n , EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
                    elif k > system.ema_n:

                        dic_au[i].loc[k,'ev_p'] = dic_au[i].loc[k-1,'ev_p'] +(mu *(dic_au[i].loc[k,'au_p']-dic_au[i].loc[k-1,'ev_p']))
                        dic_au[i].loc[k, 'ev_d'] = dic_au[i].loc[k-1,'ev_d'] +(mu *(dic_au[i].loc[k,'au_d']-dic_au[i].loc[k-1,'ev_d']))


            ################  make a logit model for agents' model choice by using predicted utility (for each agent)
            # 当drivers 的概率
            Pr = (math.exp( dic_au[i].loc[k, 'ev_d'])) / (math.exp(dic_au[i].loc[k, 'ev_d']) + math.exp(dic_au[i].loc[k, 'ev_p']))

            if flip(Pr):
                # the result of logit model should be stored in  au_P_D dataframe and new_df
                dic_au[i].loc[k,'P_D'] = 'D'
                new_df.loc[new_df[new_df['ID'] == i]['P_D'].index.values[0],'P_D']= 'D'
            else:
                dic_au[i].loc[k, 'P_D'] = 'P'
                new_df.loc[new_df[new_df['ID'] == i]['P_D'].index.values[0], 'P_D'] = 'P'

        # 追加其他column (并赋予初始值)
        new_df['partner'] = '-'  # lamp added partner# 初始值是’-’
        new_df['WaitingTime'] = 0  # lamp added matching time
        new_df['choice?'] = 0
        new_df['mode'] = 0
        new_df['rd_v'] = 0
        new_df['route'] = 0
        new_df['travel'] = 0
        new_df['detour time'] = 0
        new_df['pick up time'] = 0
        new_df['trip_time'] = 0  # store total trip time based on different agent
        new_df['toll'] = 0
        new_df['get_out_o1'] = 0
        new_df['get_out_1d'] = 0
        new_df['get_out_o2'] = 0
        new_df['get_out_2d'] = 0
        new_df['get_out_21'] = 0
        new_df['travel time_link O-1'] = 0
        new_df['travel time_link 1-D'] = 0
        new_df['travel time_link O-2'] = 0
        new_df['travel time_link 2-D'] = 0
        new_df['travel time_link 2-1'] = 0
        new_df['travel time_route 1'] = 0
        new_df['travel time_route 2'] = 0
        new_df['travel time_route 3'] = 0
        new_df['utility_ridesharing_route 1'] = 0
        new_df['utility_ridesharing_route 2'] = 0
        new_df['utility_ridesharing_route 3'] = 0
        new_df['utility_drive-alone_route 1'] = 0
        new_df['utility_drive-alone_route 2'] = 0
        new_df['utility_drive-alone_route 3'] = 0
        new_df['utility_rideshare'] = 0
        new_df['utility_drive-alone'] = 0
        new_df['P_ridesharing_route 1'] = 0
        new_df['P_ridesharing_route 2'] = 0
        new_df['P_ridesharing_route 3'] = 0
        new_df['Random_RD_route_choice'] = 0
        new_df['P_drive-alone_route 1'] = 0
        new_df['P_drive-alone_route 2'] = 0
        new_df['P_drive-alone_route 3'] = 0
        new_df['Random_DA_route_choice'] = 0
        new_df['P_rideshare'] = 0
        new_df['P_drive-alone'] = 0
        new_df['Random_mode choice'] = 0
        new_df['n_o1'] = 0
        new_df['n_1d'] = 0
        new_df['n_o2'] = 0
        new_df['n_2d'] = 0
        new_df['n_21'] = 0
        new_df['n_rideshare'] = 0
        new_df['n_drivealone'] = 0

        return new_df

    ##make a function to store result fo day k

    #############################################################################################################
    #####################CORE CODE###############################################################################
    #############################################################################################################
    # make a dictionary to store everyday results
    # dic is the final dictionary what I want. It should be define outside of the core function

    dic = {} #做一个空字典

    def sim_make_dic(system):
        # make a DataFrame to store that (include 'ev_p','ev_d','new_P_D' ). it will be used to make a diction 'dic_ac'
        #au_P_D = DataFrame(0, columns=['au_p', 'au_d','ev_p','ev_d','P_D'], index=[i for i in range(1, system.day + 1)])
        # make a dictionary to store each ID agents' result of au_P_D. it will be used in update_P_D function to get updated P_D mode
        dic_au = dict(zip([i for i in range(system.N)],[DataFrame(0, columns=['au_p', 'au_d','ev_p','ev_d','P_D'], index=[i for i in range(1, system.day + 1)]) for i in range(system.N)] ))
        #print(dic_au)
        # make a day k loop (1~N)
        hist_p = np.random.normal(loc=-5.0, scale=0, size=1000)
        hist_d = np.random.normal(loc=-5.0, scale=0, size=1000)
        #print(hist_p,hist_d)
        for k in range(1, system.day+1): # loop for day-time step in simulation
            # update_P_D function can generate 1st day mode choice by giving initial value (ac_p=ev_p, ac_d =ev_d )
            df = update_P_D(k,system,dic_au,hist_p,hist_d)
               #每天更新选择P还是D,df=datafram
            #store _result function is going to use 'run_simulation'function which uses update_func(df).
            # this function will get the latter half of result DataFame then combine the first half(df at that day k time)
            print('day: {}'.format(k))
            result = store_result(df)
            # put the dat k result into the dictionary dic (so dic should not be inside of this sim_make_dic function)

            # after the time step of system.T(means there wil not be any agents appearing),
            # we should check is there any waiting passengers, then give them (system.T - appear_time)
            left_P_index = (result.loc[(result.P_D == 'P') & (result.partner == '-')]).index.values
            MAX = result['travel'].max() #如果有乘客没有被match的时候，将其旅行时间赋予为当天最大的旅行时间

            for i in left_P_index:
                #print(i)
                result.loc[i, 'WaitingTime'] = system.T - i# OK
                result.loc[i, 'travel'] = MAX
                result.loc[i, 'trip_time'] = system.T - i + MAX #trip time包含等待时间，匹配时间和旅行时间，所以这里除了旅行时间还要加上乘客等待时间=仿真最后时间-乘客出现时的时间

            dic[k] =result #k代表天数

        #for sp_val in dic_au.values():
            #print(sp_val)
        #print(dic_au) # should output all agents' au_P_D  results{1:□,2:□........systen.N:□}
        return dic

    sim_make_dic = sim_make_dic(system)  # should be dictionary looped in day k
    #print(type(sim_make_dic))
    #print(sim_make_dic)

    matome = DataFrame(0, index=[
    'total trip time',
    'ridesharing drivers',
    'drive-alone drivers',
    'ratio of ridesharing',
    'passengers who failed in matching',
    'number of vehicle on link O-1',
    'number of vehicle on link 1-D',
    'number of vehicle on link O-2',
    'number of vehicle on link 2-1',
    'number of vehicle on link 2-D',
    'tolled vehicle',
    'Total revenue',
    'Total general cost',
    'average travel time on route 1',
    'variance travel time on route 1',
    'average travel time on route 2',
    'variance travel time on route 2',
    'average travel time on route 3',
    'variance travel time on route 3',
    'drivers',
    'passengers',
    'travel_time',
    'waiting_time',
    'detour_time',
    'average pick up time',
    'drive alone vehicle on route 1',
    'drive alone vehicle on route 2',
    'drive alone vehicle on route 3',
    'ridesharing vehicle on route 1',
    'ridesharing vehicle on route 2',
    'ridesharing vehicle on route 3'], columns=[i for i in range(1, system.day + 1)])

    ## path3 = './resources/m2.csv'

    sheet_id = 1

    for i in range(1,system.day+1):

        # 调出每天的结果，然后进行总结计算
        result = sim_make_dic[i]

        #meterics の結果をリストに変換
        metrics = [k for k in metrics_update_func(system, result)]

        #store the result of the ith day to number i row of matome(dataframe)
        matome[i]=metrics


    with pd.ExcelWriter('./resources/e{}.xls'.format(system.toll)) as writer:
        for i in range(1,system.day+1):

            sim_make_dic[i].to_excel(writer, sheet_name = '{}'.format(sheet_id))
            sheet_id += 1

    ######
    # print(sim_make_dic_write[4])
    #sim_make_dic[1].to_csv('./resources/rideshare_included_4.csv')


    matome.to_csv('./resources/m{}.csv'.format(system.toll))
        # with pd.ExcelWriter(path1) as writer:
            # writer.book = opx.load_workbook(path1)
            # sim_make_dic[i].to_excel(writer, sheet_name='1')


#　END #####################################

t2 = time.time()

elapsed_time = t2-t1
print('elapsed time:')
print(elapsed_time)

