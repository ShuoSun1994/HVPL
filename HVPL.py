import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from benchmark import *
from simple import *
import math 
from hybrid import *
from composition import *



# 设置常量
# lb = -100 # 变量的下界
# ub = 100  # 变量的上界
times = 4
fallrate = 0.05
TransportationRate = 0.36
par = 1
g = 2
maxit = 10000  # Number of iterations
# maxit = 4000  # Number of iterations
Leaguesize = 40
# nPlayer = 10
NumberOfFall = int(np.ceil(fallrate * Leaguesize))
NumberOfTransportationTeam = int(np.ceil(TransportationRate * Leaguesize))
o = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Shift vector
M = np.eye(10)  # Rotation matrix
cycle  = math.ceil(maxit / 25)
degree = 3
itpercycle = maxit / cycle
qtcycle = itpercycle / 4
alpha = 1
evaporation_rate = 0.5
P = 0.5
# stepsieze = np.zeros(nPlayer)
operators_count = np.zeros(6)
method_counts = [1, 1, 1, 1,1, 1, 1]
 
    

# test function 
def sphere(x,o,M):
    return sum([xi**2 for xi in x])

    

# compose  teams 
class Teams:
    def __init__(self,lb,ub,nPlayer,N,NumberOfFall,NumberOfTransportationTeam, function):
        self.lb = lb
        self.ub = ub
        self.nPlayer = nPlayer
        self.NumberOfFall = NumberOfFall
        self.NumberOfTransportationTeam = NumberOfTransportationTeam
        self.formation = None
        self.subsititude = None
        self.cost = None
        self.N = N
        self.pheromone_matrix = np.ones((N, nPlayer))
        self.teams = self.generate_league_teams(function)

    def initial_a_team(self,function):
        self.formation = np.random.uniform(low=self.lb, high=self.ub, size=(nPlayer,1))
        self.subsititude = np.random.uniform(low=self.lb, high=self.ub, size=(nPlayer,1))
        self.cost = function(self.formation, o,M)
        return self.formation,self.subsititude,self.cost
        
    # league teams
    def generate_league_teams(self,function):
        teams = np.array([self.initial_a_team(function) for t in np.arange(self.N)])
        return teams


    def Update_the_pheromone(self):
    # Update the pheromone matrix
        for ant in range(self.N):
            for i in range(self.nPlayer):
                self.pheromone_matrix[ant][i] *= evaporation_rate
                self.pheromone_matrix[ant][i] += (1 - evaporation_rate) / self.teams[ant][2]




# league time table
def timetable(Leaguesize):
    timetable = np.vstack((np.arange(Leaguesize),np.zeros(Leaguesize))).T
    bisectedlist = np.vstack((np.arange(int(Leaguesize / 2)), np.arange(int(Leaguesize / 2),Leaguesize)[::-1]))
    for i in np.arange(Leaguesize - 1):
        for j in np.arange(int(Leaguesize / 2) ):
            timetable[int(bisectedlist[0,j]),-1] = bisectedlist[1,j]
            timetable[int(bisectedlist[1,j]),-1] = bisectedlist[0,j]
        timetable = np.vstack((timetable.T,np.zeros(Leaguesize))).T
        temprar = np.zeros((2,int(Leaguesize / 2)))
        temprar[0,0] = 0
        temprar[0,1] = bisectedlist[1,0]
        for k in np.arange(2,int(Leaguesize / 2)):
            temprar[0,k] = bisectedlist[0, k - 1]
        for l in np.arange(int(Leaguesize / 2 - 1)):
            temprar[1,l] = bisectedlist[1, l + 1]
        temprar[1,-1] = bisectedlist[0, -1]  
        bisectedlist =  temprar
    timetable = np.delete(np.delete(timetable, -1, axis= 1), 0, axis= 1)
    return timetable
            
# Strategy 
class ApplyStrategy:
    def __init__(self, teams, team1, team2, Best_team,function,A,B,itr, CF,RL,RB):
        self.teams = teams
        self.team1 = team1
        self.team2 = team2
        self.Best_team = Best_team
        self.function = function
        self.A = A
        self.B = B
        self.itr = itr
        self.CF = CF
        self.RL = RL
        self.RB = RB
        # self.result = self.manage_strategy()
        self.result = self.selecct_strategy()

    def update_counts(self,method_counts, selected_method):
        method_counts[selected_method - 1] += 1
    
    def select_method(self):
        def calculate_weights(method_counts):
            total_counts = sum(method_counts)
            weights = [count / total_counts for count in method_counts]
            return weights  
        
        weights = calculate_weights(method_counts)
        selected_method1 = random.choices(range(1, 8), weights=weights, k=3)[1]
        return selected_method1
    
        
    def selecct_strategy(self):
        # team1
        selected_method1 = self.select_method()
        selected_method1 = 1
        if selected_method1 == 3:
            self.strategy1_kt()
        elif selected_method1 == 2:
            self.strategy1_map()
        elif selected_method1 == 1:
            self.strategy1_alns()
        elif selected_method1 == 4:
            self.strategy1_aco()
        elif selected_method1 == 5:
            self.strategy1_rp()
        elif selected_method1 == 6:
            self.strategy1_su()
        elif selected_method1 == 7:
            self.strategy1_toB()            
        if self.team1[2] < self.Best_team[2]:
            self.Best_team = self.team1
            self.update_counts(method_counts,selected_method1)
        # team2
        selected_method2 = self.select_method()
        selected_method2 = 2
        if selected_method2 == 3:
            self.strategy2_kt()
        elif selected_method2 == 2:
            self.strategy2_map()
        elif selected_method2 == 1:
            self.strategy2_alns()
        elif selected_method2 == 4:
            self.strategy2_aco()
        elif selected_method2 == 5:
            self.strategy2_rp()
        elif selected_method2 == 6:
            self.strategy2_su()
        elif selected_method2 == 7:
            self.strategy2_toB()            
        if self.team2[2] < self.Best_team[2]:
            self.Best_team = self.team2
            self.update_counts(method_counts,selected_method2)

        return [self.team1,self.team2, self.Best_team]









                      
    def manage_strategy(self):
        # # team1 
        # self.strategy1_kt()
        # #  self.strategy1_lio()
        # if self.team1[2] > self.team2[2]:
        #     self.strategy1_rp()
        #     # self.strategy1_kt()
        #     if self.team1[2] > self.team2[2]:
        #         self.strategy1_su()        

        # team1 
        # self.strategy1_kt()
        # self.strategy1_alns()
        self.strategy1_map()
        self.strategy2_map()
        # self.strategy1_aco()
        if self.team1[2] > self.team2[2]:
            self.strategy1_alns()
            self.strategy2_aco()
        #     if random.random() < 0.5:
        # #     # self.strategy1_rp()
        #         self.strategy1_aco()
        #         # self.strategy1_kt()
        # #     # self.strategy_map()
        #     else:
        #         # self.strategy_aco()
        #         self.strategy1_alns()
        # #     if self.team1[2] > self.team2[2]:
        # #         # self.strategy1_su()
        # #         # self.strategy1_kt()
        # #         self.strategy1_alns()
        else:
            self.strategy1_kt()
            self.strategy2_alns()
            # self.strategy_aco()
                 
        # self.team2[0] = np.random.rand((len(self.team2[0]))).reshape(self.Best_team[0].shape) * self.Best_team[0] + self.team2[0]
        # self.team2[2] = self.function(self.team2[0] , o,M)
        if self.team1[2] < self.Best_team[2]:
             self.Best_team = self.team1
        if self.team2[2] < self.Best_team[2]:
             self.Best_team = self.team2
             
         
        return [self.team1,self.team2, self.Best_team]
         
             
    
    def simplebound1(self):
        lb  = self.teams.lb
        ub  = self.teams.ub
        Flag4ub = self.team1[0] > ub
        Flag4lb = self.team1[0] < lb
        self.team1[0] = self.team1[0] * (~(Flag4ub + Flag4lb)) + ub * Flag4ub + lb * Flag4lb
        Flag4ub_ = self.team1[1] > ub
        Flag4lb_ = self.team1[1] < lb
        self.team1[1] = self.team1[1] * (~(Flag4ub_ + Flag4lb_)) + ub * Flag4ub_ + lb * Flag4lb_
        self.team1[2] = self.function(self.team1[0], o,M)
        
    def simplebound2(self):
        lb  = self.teams.lb
        ub  = self.teams.ub
        Flag4ub = self.team2[0] > ub
        Flag4lb = self.team2[0] < lb
        self.team2[0] = self.team2[0] * (~(Flag4ub + Flag4lb)) + ub * Flag4ub + lb * Flag4lb
        Flag4ub_ = self.team2[1] > ub
        Flag4lb_ = self.team2[1] < lb
        self.team2[1] = self.team2[1] * (~(Flag4ub_ + Flag4lb_)) + ub * Flag4ub_ + lb * Flag4lb_
        self.team2[2] = self.function(self.team2[0], o,M)      
    
    
    def strategy1_toB(self):
        self.team1[0] = np.random.rand((len(self.team1[0]))).reshape(self.Best_team[0].shape) * self.Best_team[0] + self.team1[0]
        # self.team1[2] = self.function(self.team1[0] , o,M)  
        self.simplebound1()

    def strategy2_toB(self):
        self.team2[0] = np.random.rand((len(self.team2[0]))).reshape(self.Best_team[0].shape) * self.Best_team[0] + self.team2[0]
        # self.team2[2] = self.function(self.team2[0] , o,M)
        self.simplebound2()
                   
    def strategy1_rp(self):
        int_r = int(np.random.uniform(0,self.teams.nPlayer))
        d = random.sample(range(self.teams.nPlayer),int_r)
        for i in d:
            index = int(np.random.uniform(0,self.teams.nPlayer))
            a = self.team1[0][i]
            b = self.team1[0][index]
            self.team1[0][i] = b
            self.team1[0][index] = a
        
        self.team1[2] = self.function(self.team1[0], o,M)
        
        
    def strategy1_su(self):
        int_r = int(np.random.uniform(0,self.teams.nPlayer))
        d = random.sample(range(self.teams.nPlayer),int_r)
        for i in d:
            index = int(np.random.uniform(0,self.teams.nPlayer))
            a = self.team1[0][i]
            b = self.team1[1][index]
            self.team1[0][i] = b
            self.team1[1][index] = a
        
        self.team1[2] = self.function(self.team1[0], o,M)
        
    def strategy2_rp(self):
        int_r = int(np.random.uniform(0,self.teams.nPlayer))
        d = random.sample(range(self.teams.nPlayer),int_r)
        for i in d:
            index = int(np.random.uniform(0,self.teams.nPlayer))
            a = self.team2[0][i]
            b = self.team2[0][index]
            self.team2[0][i] = b
            self.team2[0][index] = a
        
        self.team2[2] = self.function(self.team2[0], o,M)
        
        
    def strategy2_su(self):
        int_r = int(np.random.uniform(0,self.teams.nPlayer))
        d = random.sample(range(self.teams.nPlayer),int_r)
        for i in d:
            index = int(np.random.uniform(0,self.teams.nPlayer))
            a = self.team2[0][i]
            b = self.team2[1][index]
            self.team2[0][i] = b
            self.team2[1][index] = a
        
        self.team2[2] = self.function(self.team2[0], o,M)        
    
    
    def strategy1_kt(self):
        # self.team1[0] = self.team1[0] + np.random.uniform(low=self.teams.lb, high=self.teams.ub, size=(self.teams.nPlayer,1))
        self.team1[0] = self.team1[0] + np.random.uniform(low=self.teams.lb, high=self.teams.ub, size=(self.teams.nPlayer,1))*(1-np.random.uniform(low=self.teams.lb, high=self.teams.ub, size=(self.teams.nPlayer,1)))
        self.simplebound1()

    def strategy2_kt(self):
        # self.team1[0] = self.team1[0] + np.random.uniform(low=self.teams.lb, high=self.teams.ub, size=(self.teams.nPlayer,1))
        self.team2[0] = self.team2[0] + np.random.uniform(low=self.teams.lb, high=self.teams.ub, size=(self.teams.nPlayer,1))*(1-np.random.uniform(low=self.teams.lb, high=self.teams.ub, size=(self.teams.nPlayer,1)))
        self.simplebound2()
        
    def strategy1_alns(self):
        p =  random.random()
        p1 = ( 1 - itr / maxit )
        p2 = p1 + (1 - p1) / 2
        r = (2 * random.random() - 1 )
        index =  random.randint(0, len(teams.teams) - 1)
        colleagues = teams.teams[index]
        if p <= p1:
             self.team1[0] = self.team1[0]
        elif p >= p1 and p <= p2:
             D = abs(self.Best_team[0] - self.team1[0])
             self.team1[0] = self.Best_team[0] + r *  gamma * D
             
        else:
            if colleagues[2] < self.team1[2]:
                D = abs(colleagues[0] - self.team1[0])
                self.team1[0] = colleagues[0] + r * gamma * D
            else:
                D = abs(colleagues[0] - self.team1[0])
                self.team1[0] = self.team1[0] + r * gamma * D
        self.simplebound1()       
  
    def strategy2_alns(self):
        p =  random.random()
        p1 = ( 1 - itr / maxit )
        p2 = p1 + (1 - p1) / 2
        r = (2 * random.random() - 1 )
        index =  random.randint(0, len(teams.teams) - 1)
        colleagues = teams.teams[index]
        if p <= p1:
             self.team2[0] = self.team2[0]
        elif p >= p1 and p <= p2:
             D = abs(self.Best_team[0] - self.team2[0])
             self.team2[0] = self.Best_team[0] + r *  gamma * D
             
        else:
            if colleagues[2] < self.team2[2]:
                D = abs(colleagues[0] - self.team2[0])
                self.team2[0] = colleagues[0] + r * gamma * D
            else:
                D = abs(colleagues[0] - self.team2[0])
                self.team2[0] = self.team2[0] + r * gamma * D
        self.simplebound2()   

    def strategy1_aco(self):
        pheromone = self.teams.pheromone_matrix[self.A]
        self.team1[0] += alpha * pheromone.reshape(len(self.team1[0]),-1) * (self.teams.ub - self.teams.lb)
        self.simplebound1()  
            
    def strategy2_aco(self):
        pheromone = self.teams.pheromone_matrix[self.A]
        self.team2[0] += alpha * pheromone.reshape(len(self.team2[0]),-1) * (self.teams.ub - self.teams.lb)
        self.simplebound2()  
                   
    def strategy1_map(self):
        # for i in range(len(self.teams.nPlayer)):
        R =random.random()
        if self.itr < maxit / 3:
            stepsize = self.RB * (self.Best_team[0] - self.RB  * self.team1[0] )
            self.team1[0] = self.team1[0] + P * R * stepsize
        elif ( self.itr > maxit / 3 ) and ( self.itr < 2 * maxit / 3 ):
            if self.A > self.teams.N /2 :
                stepsize = RB * (RB * self.Best_team[0] - self.team1[0]) 
                self.team1[0] = self.Best_team[0] + P * CF * stepsize
            else:
                stepsize = RL * (self.Best_team[0] - RL * self.team1[0])
                self.team1[0] = self.team1[0] + P * R * stepsize
        else:
            stepsize = RL * ( RL *self.Best_team[0] - self.team1[0])
            self.team1[0] = self.Best_team[0] + P * R * stepsize
        self.simplebound1() 

    def strategy2_map(self):
        # for i in range(len(self.teams.nPlayer)):
        R =random.random()
        if self.itr < maxit / 3:
            stepsize = self.RB * (self.Best_team[0] - self.RB  * self.team2[0] )
            self.team2[0] = self.team2[0] + P * R * stepsize
        elif ( self.itr > maxit / 3 ) and ( self.itr < 2 * maxit / 3 ):
            if self.A > self.teams.N /2 :
                stepsize = RB * (RB * self.Best_team[0] - self.team2[0]) 
                self.team2[0] = self.Best_team[0] + P * CF * stepsize
            else:
                stepsize = RL * (self.Best_team[0] - RL * self.team2[0])
                self.team2[0] = self.team2[0] + P * R * stepsize
        else:
            stepsize = RL * ( RL *self.Best_team[0] - self.team2[0])
            self.team2[0] = self.Best_team[0] + P * R * stepsize
        self.simplebound2() 

    
# Competition
def competition(A, B, team, Best_team, function,itr):
    mincost = np.min(team.teams[:,-1])
    maxcost = np.max(team.teams[:,-1])
    m = np.zeros(len(team.teams))
    for i in np.arange(len(team.teams)):
        if maxcost > mincost:
            m[i] = (team.teams[i,2] - maxcost) / (mincost - maxcost)

    MA = m[A] / np.sum(m)
    MB = m[B] / np.sum(m)
    x = np.copy(team.teams[A])
    y = np.copy(team.teams[B])
    d = MA / (MA + MB)
    r = np.random.rand()
    if d > r :
        result1 = ApplyStrategy(team, x, y, Best_team, function,A,B,itr, CF,RL,RB).result
        X, Y ,new_Best_team = result1
        if X[2] < x[2]:
           x = X
           
        if Y[2] < y[2]:
               y = Y   
    else:
        result2 = ApplyStrategy(team,y, x, Best_team, function,A,B,itr, CF,RL,RB).result
        Y, X, new_Best_team = result2
        if Y[2] < y[2]:
               y = Y    
               
        if X[2] < x[2]:
               x = X   
    return x, y, new_Best_team
    

def simplebound_U(lb,ub,team,function):
        Flag4ub = team[0] > ub
        Flag4lb = team[0] < lb
        team[0] = team[0] * (~(Flag4ub + Flag4lb)) + ub * Flag4ub + lb * Flag4lb
        Flag4ub_ = team[1] > ub
        Flag4lb_ = team[1] < lb
        team[1] = team[1] * (~(Flag4ub_ + Flag4lb_)) + ub * Flag4ub_ + lb * Flag4lb_
        team[2] = function(team[0], o,M)
        return team


class NewTeam(Teams):
    def __init__(self,teams,lb,ub,nPlayer,Leaguesize,NumberOfFall,NumberOfTransportationTeam,function):
        super(NewTeam, self).__init__(lb,ub,nPlayer,Leaguesize,NumberOfFall,NumberOfTransportationTeam,function)
        self.n = len(teams.teams)
        self.NumberOfFall = teams.NumberOfFall
        self.formation = np.zeros((teams.nPlayer,1))
        self.subsititude = np.zeros((teams.nPlayer,1))
        self.cost = None
        self.teams = self.generate_new_teams(function)
        
        
    
    def initial_a_newteam(self,function):
        for i in np.arange(self.NumberOfFall):
            self.formation[i] = teams.teams[np.random.randint(self.n)][0][i]
            self.subsititude[i] = teams.teams[np.random.randint(self.n)][1][i]
        self.cost = function(self.formation, o,M)
        return self.formation,self.subsititude,self.cost
    
    def generate_new_teams(self,function):
        new_teams = np.array([self.initial_a_newteam(function) for t in np.arange(self.NumberOfFall)])
        return new_teams
    
    
def plot_iteration_result(result,maxit,name,num):
    result_array = np.array(result)
    x_axis = np.arange(maxit)
    plt.figure(figsize=(10,8))
    plt.plot(x_axis, result_array,color = 'red', label= name)
    plt.xlabel('epoch',fontsize=20)
    plt.ylabel('cost',fontsize=20)
    plt.title(name,fontsize=20)
    plt.xlim(0,1000)
    # plt.ylim(0,10000)
    # plt.xticks(np.arange(7850740,7250760,2))
    # plt.yticks(np.arange(7850740,7250760,2))
    plt.legend()
    plt.savefig(r'C:\\Users\\sunsh\\Desktop\\code\\test1\\picture\\{}_{}.png'.format(name,num))
    # plt.show()    
        

def  levy(nPlayer, beta):
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den = math.gamma((1+beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sifmau = (num / den) ** (1 / beta)
    u = np.random.uniform(low=0.0, high=sifmau, size=(nPlayer,1))
    v = np.random.uniform(low=0.0, high=1.0, size=(nPlayer,1))
    z = u / (abs(v) **  (1 /beta))
    return z


    




if __name__ == "__main__":
    test_set1 = ['f1', (-100,100), 50, sphere]
    test_set2 = ['f2', (-1.28,1.28), 50, quartic_noise]
    test_set3 = ['f3', (-4,5), 50, powell_singular]
    # test_set4 = ['f4', (-10,10), 50, dixon_price_function]
    test_set4 = ['f4', (-5,10), 50, zakharov]
    test_set5 = ['f5', (-10,10), 50, schwefel_2_23]
    # test_set6 = ['f6', (-4.5,4.5), 2, beale]
    test_set6 = ['f6', (-5,5), 2, three_hump_camel]
    test_set7 = ['f7', (-10,10), 2, brent]
    test_set8 = ['f8', (-10,10), 2, booth]
    test_set9 = ['f9', (-1.2,1.2), 2, leon]
    test_set10 = ['f10', (-10,10), 2, matyas]
    test_set11 = ['f11', (-500,500), 50, schwefel_226]
    test_set12 = ['f12', (-5.12,5.12), 50, rastrigin]
    test_set13 = ['f13', (-10,10), 50, periodic]
    test_set14 = ['f14', (-32,32), 50, ackley]
    test_set15 = ['f15', (-10,10), 50, alpine_n1]
    # test_set16 = ['f16', (-5,5), 2, egg_crate]
    test_set16 = ['f16', (-5,10), 2, branin_rcos]
    test_set17 = ['f17', (-10,10), 2, cross_in_tray]
    test_set18 = ['f18', (0, 1), 6, hartman6]
    test_set19 = ['f19', (-2*np.pi,2*np.pi), 2, bird]
    test_set20 = ['f20', (-5,5), 2, camel_six_hump]
    test_set21 = ['f21', (-100,100), 10, shifted_rotated_bent_cigar] 
    test_set22 = ['f22', (-100,100), 10, shifted_rotated_rosenbrock] 
    test_set23 = ['f23', (-100,100), 10, shifted_rotated_rastrigin] 
    test_set24 = ['f24', (-100,100), 10, shifted_rotated_expanded_scaffer] 
    test_set25 = ['f25', (-100,100), 10, shifted_rotated_lunacek_bi_rastrigin] 
    test_set26 = ['f26', (-100,100), 10, shifted_rotated_non_continuous_rastrigin] 
    test_set27 = ['f27', (-100,100), 10, shifted_rotated_levy] 
    test_set28 = ['f28', (-100,100), 10, shifted_rotated_schwefel] 
    test_set29 = ['f29', (-100,100), 10, f11]
    test_set30 = ['f30', (-100,100), 10, f13] 
    test_set31 = ['f31', (-100,100), 10, f14]
    test_set32 = ['f32', (-100,100), 10, f15]
    test_set33 = ['f33', (-100,100), 10, f22]   
    test_set34 = ['f34', (-100,100), 10, f26]
    test_set35 = ['f35', (-100,100), 10, f28]
    # test_set21 = ['f21', (-100,100), 10, f1] 
    # test_set22 = ['f22', (-100,100), 10, f2] 
    # test_set23 = ['f23', (-100,100), 10, f3] 
    # test_set24 = ['f24', (-100,100), 10, f4] 
    # test_set25 = ['f25', (-100,100), 10, f5] 
    # test_set26 = ['f26', (-100,100), 10, f6] 
    # test_set27 = ['f27', (-100,100), 10, f7] 
    # test_set28 = ['f28', (-100,100), 10, f8] 
    
    # test_set29 = ['f29', (-100,100), 10, f11]   
    # test_set30 = ['f30', (-100,100), 10, f12] 
    # test_set31 = ['f31', (-100,100), 10, f13] 
    # test_set32 = ['f32', (-100,100), 10, f14]
    # test_set33 = ['f33', (-100,100), 10, f15] 
    # test_set34 = ['f34', (-100,100), 10, f16] 
    # test_set35 = ['f35', (-100,100), 10, f17] 
    # test_set36 = ['f36', (-100,100), 10, f18] 
    # test_set37 = ['f37', (-100,100), 10, f19] 
    # test_set38 = ['f38', (-100,100), 10, f20]  

    # test_set39 = ['f39', (-100,100), 10, f21]   
    # test_set40 = ['f40', (-100,100), 10, f22] 
    # test_set41 = ['f41', (-100,100), 10, f23] 
    # test_set42 = ['f42', (-100,100), 10, f24]
    # test_set43 = ['f43', (-100,100), 10, f25] 
    # test_set44 = ['f44', (-100,100), 10, f26] 
    # test_set45 = ['f45', (-100,100), 10, f27] 
    # test_set46 = ['f46', (-100,100), 10, f28] 
    # test_set47 = ['f47', (-100,100), 10, f29] 
    # test_set48 = ['f48', (-100,100), 10, f30]  
    # test_set_list = [test_set1, test_set2,test_set3,test_set4,test_set5,test_set6,test_set7,test_set8,test_set9,test_set10,test_set11,test_set12,test_set13, test_set14,test_set15,test_set16,test_set17,test_set18,test_set19,test_set20]
    # test_set_list = [test_set21,test_set22,test_set23,test_set24,test_set25,test_set26,test_set27,test_set28]
    # test_set_list = [test_set29,test_set30,test_set31,test_set32,test_set33,test_set34,test_set35,test_set36,test_set37,test_set38]
    # test_set_list = [test_set31,test_set32,test_set33,test_set34,test_set35,test_set36,test_set37,test_set38]    
    # test_set_list = [test_set39,test_set40,test_set41,test_set42,test_set43,test_set44,test_set45,test_set46,test_set47,test_set48]
    # test_set_list = [test_set1, test_set2,test_set3,test_set4,test_set5,test_set6,test_set7,test_set8,test_set9,test_set10,test_set11,test_set12,test_set13, test_set14,test_set15,test_set16,test_set17,test_set18,test_set19,test_set20]
    test_set_list = [test_set16]

    for test_set in test_set_list:
        for num in range(times):
            name = test_set[0]
            lb = test_set[1][0]
            ub = test_set[1][1]
            nPlayer = test_set[2]
            function = test_set[3]
            start_time = time.time()
            stepsieze = np.zeros(nPlayer)
            teams = Teams(lb,ub,nPlayer,Leaguesize,NumberOfFall,NumberOfTransportationTeam,function)
            best_team = teams.teams[np.argmin(teams.teams[:,-1])]
            Best_team = np.copy(best_team)
            Best_cost_list = []
        # start loop
            for itr in np.arange(maxit):
                CF = (1 - itr/maxit) ** (2 * itr / maxit)
                RL = 0.05 * levy(nPlayer,  1.5)
                RB = np.random.normal(size = (nPlayer,1))
                a = g - itr * (g / maxit)
                gamma = abs(2 - ((itr % itpercycle) + 1) / qtcycle)
                schedule = timetable(Leaguesize)
                for i in np.arange(Leaguesize - 1):
                    k = schedule[:,i]
                    for j in np.arange(Leaguesize):
                        A = int(k[j])
                        B = j

                        x, y, Best_team_ = competition(A, B, teams, Best_team, function,itr)
                        Best_team = np.copy(Best_team_)
                        teams.teams[A] = x
                        teams.teams[B] = y
                        
                    # learning phase
                    sort_index1 = np.argsort(teams.teams[:,2])
                    rank1 = teams.teams[sort_index1[0]]
                    rank2 = teams.teams[sort_index1[1]]
                    rank3 = teams.teams[sort_index1[2]]
                    for l in np.arange(Leaguesize):
                        U = np.copy(teams.teams[l]) 
                        
                        r1 = np.random.rand()
                        r2 = np.random.rand()
                        H1 = g * a * r1 - a
                        G1 = g * r2
                        Dx1 = np.abs(G1 * rank1[0] - U[0])
                        x1 = rank1[0] - Dx1 * H1
                        
                        r1 = np.random.rand()
                        r2 = np.random.rand()
                        H2 = g * r1 - a
                        G2 = g * r2
                        Dx2 = np.abs(G2 * rank2[0] - U[0])
                        x2 = rank2[0] - Dx1 * H1
                    
                        r1 = np.random.rand()
                        r2 = np.random.rand()
                        H3 = g * r1 - a
                        G3 = g * r2
                        Dx3 = np.abs(G3 * rank3[0] - U[0])
                        x3 = rank3[0] - Dx1 * H1
                        
                        U[0] = (x1 + x2 + x3) / 3
                        U = simplebound_U(lb,ub,U,function)
                        if U[2] < teams.teams[l][2] :
                            teams.teams[l] = U
                        if Best_team[2] > teams.teams[l][2]: 
                            Best_team =  teams.teams[l]
                        min_index = np.argmin(teams.teams[:,2])
                        if Best_team[2] > teams.teams[min_index][2]:
                            Best_team = teams.teams[min_index]
                # falldown
                sort_index2 = np.argsort(teams.teams[:,2])    
                teams.teams = teams.teams[sort_index2]
                # eliminate two teams
                teams.teams = np.delete(teams.teams,  np.s_[(Leaguesize - NumberOfFall):Leaguesize], axis= 0)  
                new_teams = NewTeam(teams,lb,ub,nPlayer,Leaguesize,NumberOfFall,NumberOfTransportationTeam,function)
                teams.teams = np.vstack((teams.teams,new_teams.teams))
                
                # transporation
                transportation_index = random.sample(range(Leaguesize),NumberOfTransportationTeam)
                
                for t in transportation_index:
                    S = teams.teams[t]
                    NumberOfTransportedPlayer = int(np.ceil(np.random.rand() * nPlayer))
                    NumberOfTransportedFormPlayer = int(np.ceil(np.random.rand() * NumberOfTransportedPlayer))
                    NumberOfTransportedSubsititudePlayer = NumberOfTransportedPlayer - NumberOfTransportedFormPlayer
                    SelectedForm = random.sample(range(nPlayer), NumberOfTransportedFormPlayer)
                    for q in SelectedForm:
                        a1 = teams.teams[t][0][q]
                        TeamIndex = np.random.randint(Leaguesize)
                        FormIndex = np.random.randint(nPlayer)
                        a2 = teams.teams[TeamIndex][0][FormIndex]
                        teams.teams[t][0][q] = a2
                        teams.teams[t][2] = function(teams.teams[t][0], o,M)
                        teams.teams[TeamIndex][0][FormIndex] = a1
                        teams.teams[TeamIndex][2] = function(teams.teams[TeamIndex][0], o,M)
                    SelectedSubsititude = random.sample(range(nPlayer),NumberOfTransportedSubsititudePlayer)
                    for s in SelectedSubsititude:
                        b1 = teams.teams[t][1][s]
                        TeamIndexS = np.random.randint(Leaguesize)
                        SubsititudeIndex = np.random.randint(nPlayer)
                        b2 = teams.teams[TeamIndexS][1][SubsititudeIndex]
                        teams.teams[t][1][s] = b2
                        teams.teams[TeamIndex][1][SubsititudeIndex] = b1
                    
                    if teams.teams[t][2] > S[2]:
                        teams.teams[t] = S
                
                final_sort_index = np.argsort(teams.teams[:,2])     
                teams.teams = teams.teams[final_sort_index]   
                if teams.teams[0,2] < Best_team[2]:
                    Best_team = teams.teams[0]
                
                Best_cost_list.append(Best_team[2])
                print('Best Cost of Iteration {} is {} '.format(itr, Best_team[2]))
                teams.Update_the_pheromone()
            end_time = time.time() - start_time
            print("Finished epoch , took {} s".format(time.strftime('%H:%M:%S', time.gmtime(end_time))))    
            plot_iteration_result(Best_cost_list,maxit,name,num)
            with open('results.csv', 'a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['No.{}:{} ----> {}'.format(num,name, float(Best_team[2]))])
                writer.writerow(['-------------------------------------'])
        
        
                
                
                
                
            
            
        
    
