import pandas as pd
import math
import numpy as np
###################################################################################
############### Make a class for matching process##################################
###################################################################################

class Matching:
    def __init__(self, t, df, n_rd, n_p, v_cap, N, O_x, O_y):
        self.t = t
        self.df = df
        self.n_rd = n_rd
        self.n_p = n_p
        self.mode = 'NA'
        self.partners =[]
        self.v_cap = v_cap
        self.N = N
        self.O_x = O_x
        self.O_y = O_y

    def distance_to_origin(self,id):

        id_index = (self.df.loc[self.df['ID'] == id]).index[0]

        # this function is for calculating the distance between the agents (drivers and passengers) and origin
        agent_x_location = self.df.loc[id_index, 'X-Location']
        agent_y_location = self.df.loc[id_index, 'Y-Location']

        distance_to_o = math.sqrt(math.pow(agent_x_location-self.O_x, 2) + math.pow(agent_y_location-self.O_y, 2))

        return distance_to_o

    def min_distance(self):

        self.mode = 1
        # if # of waiting passengers >= 1, then
        if self.n_p >= self.v_cap:
            self.n_p -= self.v_cap
        else: # when self.v_cap > 1, this step is necessary
            n_matched_p = self.n_p
            self.n_p -= self.n_p

        # get partners
        # self.t is the time step when doing matching process

        store_min = []
        store_ID = []
        store_distance_to_o = []

        for iter_t in range(1,self.t):
            role = self.df.loc[iter_t, 'P_D'] # DataFrame.loc[行，列]
            partner = self.df.loc[iter_t, 'partner']
            pax_id = self.df.loc[iter_t, 'ID']
            x_location = self.df.loc[iter_t, 'X-Location']
            y_location = self.df.loc[iter_t, 'Y-Location']
            D_x_location = self.df.loc[self.t, 'X-Location']
            D_y_location = self.df.loc[self.t, 'Y-Location']

            if role == 'P' and partner == '-':

                store_min.append(math.sqrt(math.pow(x_location-D_x_location,2)+math.pow(y_location-D_y_location,2)))
                store_ID.append(pax_id)
                store_distance_to_o.append(self.distance_to_origin(pax_id))
            else:
                pass

        result = []

        for i in range(len(store_min)):
            result.append(store_min[i]+store_distance_to_o[i])

        result_min = min(result)
        index_ID = result.index(result_min)

        self.partners.append(store_ID[index_ID])

        min_distance_d_p = store_min[index_ID]

        return min_distance_d_p


    def run_matching(self):
        self.n_rd += 1
        if self.n_rd > 0 and self.n_p > 0:
            self.min_distance()

        else:
            print("WARN: Matching wasn't executed due to insufficient agents")

    def get_N_rd(self):
        return self.n_rd

    def get_N_p(self):
        return self.n_p

    def get_mode(self):
        return self.mode

    def get_partners(self):
        return self.partners


def main():
    t = 3
    agentID = [11, 12, 13, 14, 15, 16]
    PorD = ['P', 'P', 'P', 'D', 'D', 'D']
    partner = ['5', '-', '-', '-', '-', '-']
    X_location = [1, 2, 3, 4, 5, 6]
    Y_location = [1, 2, 3, 4, 5, 6]
    n_rd = 0
    n_p = 3
    v_cap = 1
    N = 100
    O_x = 2
    O_y = 2
    d = {'P_D': PorD, 'ID': agentID, 'partner': partner, 'X-Location': X_location, 'Y-Location': Y_location}
    df = pd.DataFrame(d)
    ma = Matching(t, df, n_rd, n_p, v_cap, N, O_x, O_y)
    ma.run_matching()

    print('N_rd', ma.get_N_rd(), n_rd)
    print('N_p', ma.get_N_p(), n_p)
    print('partners', ma.get_partners())
    print('mode', ma.get_mode())

    print(ma.distance_to_origin(15))


# below two lines are to avoid "immediately running after import this
if __name__ == "__main__":
    main()


