import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import copy
from MDVRPPDTW import run, plotObj, plotRoutes, outPut , Comparae_best_values,compute_and_comparae_best_values,movePosition,upateTau,crossSol
# 设置常量
sb = 0.2  # 队内学习最好球队概率
fallrate = 0.15
TransportationRate = 0.36
par = 1
g = 2

maxit = 100  # Number of iterations
# Leaguesize = 10
# nPlayer = 10
Leaguesize = 100
nPlayer = 100
NumberOfFall = int(np.ceil(fallrate * Leaguesize))
NumberOfTransportationTeam = int(np.ceil(TransportationRate * Leaguesize))



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
    def __init__(self, teams_f,teams_s, team1, team2, team1_index,Best_team ,sb,x_s,y_s,team2_index):
        self.teams_f = teams_f
        self.teams_s = teams_s
        self.team1 = copy.deepcopy(team1)
        self.team2 = copy.deepcopy(team2)
        self.team1_s = x_s
        self.team2_s = y_s
        self.index = team1_index
        self.y_index = team2_index
        self.Best_team = copy.deepcopy(Best_team)
        self.sb = sb
        self.result = self.manage_strategy()
        
        
    def manage_strategy(self):
        # team1 
        self.strategy_x1()
        #  if self.team1.obj > self.team2.obj:
        #      self.strategy_x2()
        #      if self.team1.obj > self.team2.obj:
        #          self.strategy_x3()
          # team2         
        self.strategy_y()
        if self.team1.obj > self.team2.obj:   
            self.strategy_x2()
        else:
            self.strategy_y2()
            
        return [self.team1,self.team2, self.Best_team]
         
             
    
    
           
    def strategy_x2(self):
        sol_list = copy.deepcopy(self.teams_f.sol_list)
        best_team = copy.deepcopy(self.Best_team)
        index = self.index
        teams_f.sol_list[index] = None
        team = copy.deepcopy(sol_list[index])  
        int_r = int(np.random.uniform(0,len(self.teams_f.demand_id_list)))
        d = random.sample(range(len(self.teams_f.demand_id_list)),int_r)
        for i in d:
            j = int(np.random.uniform(0,len(self.teams_f.demand_id_list) ))
            if i != j:
                a = team.node_id_list[i]
                b = team.node_id_list[j]
                team.node_id_list[i]= b
                team.node_id_list[j] = a
        compute_and_comparae_best_values(self.teams_f,best_team,team)
        teams_f.sol_list[index] = (copy.deepcopy(team))
        self.team1 = teams_f.sol_list[index]
        self.Best_team = teams_f.best_sol
        
        
        
    # def strategy_x3(self):
    def strategy_y2(self):
        sol_list = copy.deepcopy(self.teams_f.sol_list)
        sol_list_s = copy.deepcopy(self.teams_s.sol_list)
        best_team = copy.deepcopy(self.Best_team)
        # index = self.index
        index = self.y_index
        teams_f.sol_list[index] = None
        teams_s.sol_list[index] = None
        team = copy.deepcopy(sol_list[index]) 
        team_s = copy.deepcopy(sol_list_s[index]) 
        int_r = int(np.random.uniform(0,len(self.teams_f.demand_id_list)))
        d = random.sample(range(len(self.teams_f.demand_id_list)),int_r)
        for i in d:
            j = int(np.random.uniform(0,len(self.teams_f.demand_id_list)))
            a = team.node_id_list[i]
            b = team_s.node_id_list[j]
            if a == b:
                team.node_id_list[i] = b
                team_s.node_id_list[j] = a
        
        compute_and_comparae_best_values(self.teams_f,best_team,team)
        teams_f.sol_list[index] = (copy.deepcopy(team))
        self.team1 = teams_f.sol_list[index]
        compute_and_comparae_best_values(self.teams_s,best_team,team_s)
        teams_s.sol_list[index] = (copy.deepcopy(team_s))
        self.team1_s = teams_s.sol_list[index]
        self.Best_team = teams_f.best_sol
        
    def strategy_y(self):
        index = self.y_index
        teams_f.sol_list[index] = None
        movePosition(self.teams_f,index)
        self.team2 = teams_f.sol_list[index]
        self.Best_team = teams_f.best_sol
    
    def strategy_x1(self):
        sol_list = copy.deepcopy(self.teams_f.sol_list)
        best_team = copy.deepcopy(self.Best_team)
        index = self.index
        teams_f.sol_list[index] = None
        team = copy.deepcopy(sol_list[index])  
        
        if random.random() <= self.sb:
            # 挑选解序列中某个位置
            cro1_index=int(random.randint(0,len(teams_f.demand_id_list)-1))
            cro2_index=int(random.randint(cro1_index,len(teams_f.demand_id_list)-1))
            new_c1_f = []  # 新的解序列前段
            new_c1_m=team.node_id_list[cro1_index:cro2_index+1]  # 新的解序列中段
            new_c1_b = []    # 新的解序列后段
            for num in range(len(teams_f.demand_id_list)):
                # 之所以分前中后 是因为 这样相当于将另外一个解序列的一段拼接上去
                if len(new_c1_f)<cro1_index:  
                    if best_team.node_id_list[num] not in new_c1_m:
                        new_c1_f.append(best_team.node_id_list[num])
                else:
                    if best_team.node_id_list[num] not in new_c1_m:
                        new_c1_b.append(best_team.node_id_list[num])
            new_c1=copy.deepcopy(new_c1_f)
            new_c1.extend(new_c1_m)
            new_c1.extend(new_c1_b)
            team.node_id_list=new_c1
            compute_and_comparae_best_values(self.teams_f,best_team,team)
            teams_f.sol_list[index] = (copy.deepcopy(team))
            self.team1 = teams_f.sol_list[index]
        else:
            teams_f.sol_list[index] = (copy.deepcopy(team))
            self.team1 = teams_f.sol_list[index]
        self.Best_team = teams_f.best_sol
    
    

    
# Competition
def competition(A, B, teams_f, Best_team,teams_s,sb):
    sol_list = copy.deepcopy(teams_f.sol_list)
    sol_list_s = copy.deepcopy(teams_s.sol_list)
    cost_list = [x.obj for x in sol_list]
    mincost = min(cost_list)
    maxcost = max(cost_list)
    m = np.zeros(len(teams_f.sol_list))
    for i in np.arange(len(teams_f.sol_list)):
        m[i] = (cost_list[i] - maxcost) / ((mincost - maxcost)+ 1*10**(-8) )

    MA = m[A] / np.sum(m)
    MB = m[B] / np.sum(m)
    x = sol_list[A]
    x_s = sol_list_s[A]
    y = sol_list[B]
    y_s = sol_list_s[B]
    d = MA / (MA + MB)
    r = np.random.rand()
    if d > r :
        result1 = ApplyStrategy(teams_f,teams_s, x, y, A,Best_team,sb,x_s,y_s,B).result
        X, Y ,new_Best_team = result1
        if X.obj < x.obj:
           x = X
           
        if Y.obj < y.obj:
               y = Y   
    else:
        result2 = ApplyStrategy(teams_f,teams_s,y, x, B, Best_team,sb,y_s,x_s,A).result
        Y, X, new_Best_team = result2
        if Y.obj < y.obj:
               y = Y    
               
        if X.obj < x.obj:
               x = X 
               
    return x, y, new_Best_team
    



        

    
    
def plot_iteration_result(result,maxit):
    result_array = np.array(result)
    x_axis = np.arange(maxit)
    plt.figure(figsize=(10,8))
    plt.plot(x_axis, result_array,color = 'red', label="Best Cost")
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.title('Best Cost')
    plt.xlim(0,100)
    # plt.ylim(0,10000)
    # plt.xticks(np.arange(7850740,7250760,2))
    # plt.yticks(np.arange(7850740,7250760,2))
    plt.legend()
    plt.show()    
        



def to_csv(best_obj_list,store_result):
    array_format = pd.DataFrame(best_obj_list)
    array_format.to_csv(store_result,header=None)  # don't write header
    
    




if __name__ == "__main__":
    # C101_30
    demand_file1 = 'E:\ss_code\VPL\data\MDVRPPDTW\C101_30.csv'
    depot_file1 = 'E:\ss_code\VPL\data\MDVRPPDTW\depot1_1.csv'
    store_result1 = 'E:\ss_code\VPL\\result\VWCA\C101_30.csv'
    # C201_50
    demand_file2 = 'E:\ss_code\VPL\data\MDVRPPDTW\C201_50.csv'
    depot_file2 = 'E:\ss_code\VPL\data\MDVRPPDTW\depot1_3.csv'
    store_result2 = 'E:\ss_code\VPL\\result\VWCA\C201_50.csv'
    # C207_100
    demand_file3 = 'E:\ss_code\VPL\data\MDVRPPDTW\C207_100.csv'
    depot_file3 = 'E:\ss_code\VPL\data\MDVRPPDTW\depot1_5.csv'
    store_result3 = 'E:\ss_code\VPL\\result\VWCA\C207_100.csv'
    # R102_30
    demand_file4 = 'E:\ss_code\VPL\data\MDVRPPDTW\R102_30.csv'
    depot_file4 = 'E:\ss_code\VPL\data\MDVRPPDTW\depot1_1.csv'
    store_result4 = 'E:\ss_code\VPL\\result\VWCA\R102_30.csv'
    # R110_100
    demand_file5 = 'E:\ss_code\VPL\data\MDVRPPDTW\R110_100.csv'
    depot_file5 = 'E:\ss_code\VPL\data\MDVRPPDTW\depot1_5.csv'
    store_result5 = 'E:\ss_code\VPL\\result\VWCA\R110_100.csv'
    # R205_50
    demand_file6 = 'E:\ss_code\VPL\data\MDVRPPDTW\R205_50.csv'
    depot_file6 = 'E:\ss_code\VPL\data\MDVRPPDTW\depot1_3.csv'
    store_result6 = 'E:\ss_code\VPL\\result\VWCA\R205_50.csv'
    # RC101_30
    demand_file7 = 'E:\ss_code\VPL\data\MDVRPPDTW\RC101_30.csv'
    depot_file7 = 'E:\ss_code\VPL\data\MDVRPPDTW\depot1_1.csv'
    store_result7 = 'E:\ss_code\VPL\\result\VWCA\RC101_30.csv'
    # RC203_50
    demand_file8 = 'E:\ss_code\VPL\data\MDVRPPDTW\RC203_50.csv'
    depot_file8 = 'E:\ss_code\VPL\data\MDVRPPDTW\depot1_3.csv'
    store_result8 = 'E:\ss_code\VPL\\result\VWCA\RC203_50.csv'
    # RC204_100
    demand_file9 = 'E:\ss_code\VPL\data\MDVRPPDTW\RC204_100.csv'
    depot_file9 = 'E:\ss_code\VPL\data\MDVRPPDTW\depot1_5.csv'
    store_result9 = 'E:\ss_code\VPL\\result\VWCA\RC204_100.csv'
    
    file_tuple1 = (demand_file1,depot_file1,store_result1)
    file_tuple2 = (demand_file2,depot_file2,store_result2)
    file_tuple3 = (demand_file3,depot_file3,store_result3)
    file_tuple4 = (demand_file4,depot_file4,store_result4)
    file_tuple5 = (demand_file5,depot_file5,store_result5)
    file_tuple6 = (demand_file6,depot_file6,store_result6)
    file_tuple7 = (demand_file7,depot_file7,store_result7)
    file_tuple8 = (demand_file8,depot_file8,store_result8)
    file_tuple9 = (demand_file9,depot_file9,store_result9)
    
    file_list = [file_tuple1,file_tuple2,file_tuple3,file_tuple4,file_tuple5,file_tuple6,file_tuple7,file_tuple8,file_tuple9]
    
    for demand_file,depot_file,store_result in  file_list:
        start_time = time.time()
        teams_f = run(demand_file=demand_file,depot_file=depot_file,store_result = store_result,popsize=Leaguesize,v_cap=80,v_speed=1,opt_type=0,n_select=Leaguesize - NumberOfFall,pc=0.5,Q=10,tau0=10,alpha=1,beta=5,rho=0.1)
        # teams = Teams(lb,ub,nPlayer,Leaguesize,NumberOfFall,NumberOfTransportationTeam,function)
        # best_team = teams.teams[np.argmin(teams.teams[:,-1])]
        best_team = Comparae_best_values(teams_f)
        Best_team = copy.deepcopy(best_team)
        history_best_obj = [] 
        # start loop
        # for itr in np.arange(maxit):
        teams_s = run(demand_file=demand_file,depot_file=depot_file,store_result = store_result,popsize=Leaguesize,v_cap=80,v_speed=1,opt_type=0,n_select=Leaguesize - NumberOfFall,pc=0.5,Q=10,tau0=10,alpha=1,beta=5,rho=0.1)
        # 计算替补球队的cost
        _ = Comparae_best_values(teams_s)
        # a = g - itr * (g / maxit)
        schedule = timetable(Leaguesize)
        local_result_list = []
        for i in np.arange(Leaguesize - 1):
            a = g - i * (g / Leaguesize)
            k = schedule[:,i]
            for j in np.arange(Leaguesize):
                A = int(k[j])
                B = j

                x, y, Best_team_ = competition(A, B, teams_f, Best_team,teams_s,sb)
                Best_team = copy.deepcopy(Best_team_)
                teams_f.sol_list[A] = x
                teams_f.sol_list[B] = y
                
            # learning phase
            sort_index1 = np.argsort(np.array([x.obj for x in teams_f.sol_list]))
            rank1 = copy.deepcopy(teams_f.sol_list[sort_index1[0]])
            rank2 = copy.deepcopy(teams_f.sol_list[sort_index1[1]])
            rank3 = copy.deepcopy(teams_f.sol_list[sort_index1[2]])
            rank_dic = {i:x for i,x in enumerate([rank1, rank2,rank3])}
            sol_list_copy=copy.deepcopy(teams_f.sol_list)
            teams_f.sol_list=[] 
            for l in np.arange(Leaguesize):
                rank_index = random.randint(0,len(rank_dic) - 1)
                learner = rank_dic[rank_index]  # 学习的对象
                best_team = copy.deepcopy(teams_f.best_sol)
                U = copy.deepcopy(sol_list_copy[l])
                front = random.randint(0,len(sol_list_copy)-2)
                back = random.randint(front + 1,len(sol_list_copy)-1)
                copy_location = random.randint(1,3)
                if copy_location == 1:  # 前段
                    new_U_f = []  
                    new_U_m_b =U.node_id_list[front:len(teams_f.demand_id_list)]
                    for id in range(len(teams_f.demand_id_list)):
                        if len(new_U_f)<front:
                            if learner.node_id_list[id] not in new_U_m_b:
                                new_U_f.append(learner.node_id_list[id])
                    new_U = copy.deepcopy(new_U_f) 
                    new_U.extend(new_U_m_b)
                    U.node_id_list = new_U
                if copy_location == 2:   # 中段
                    new_U_m = []  
                    new_U_f =U.node_id_list[0:front]
                    new_U_b =U.node_id_list[back:len(teams_f.demand_id_list)]
                    new_U_f_b = new_U_f + new_U_b
                    for id in range(len(teams_f.demand_id_list)):
                        if len(new_U_m)<(back - front):
                            if learner.node_id_list[id] not in new_U_f_b:
                                new_U_m.append(learner.node_id_list[id])
                    new_U = new_U_f 
                    new_U.extend(copy.deepcopy(new_U_m))
                    new_U.extend(new_U_b)
                    U.node_id_list = new_U
                if copy_location == 3:   # 后段
                    new_U_b= []  
                    new_U_f_m =U.node_id_list[0:back]
                    for id in range(len(teams_f.demand_id_list)):
                        if len(new_U_b)<(len(teams_f.demand_id_list) - back ):
                            if learner.node_id_list[id] not in new_U_f_m:
                                new_U_b.append(learner.node_id_list[id])
                    new_U = new_U_f_m
                    new_U.extend(copy.deepcopy(new_U_b))
                    U.node_id_list = new_U
                compute_and_comparae_best_values(teams_f,best_team,U)
                if U.obj < sol_list_copy[l].obj:
                    teams_f.sol_list.append(copy.deepcopy(sol_list_copy[l]))
                else:
                    teams_f.sol_list.append(copy.deepcopy(U))
            local_result_list.append(teams_f.best_sol.obj)    
            if len(local_result_list) - len(set(local_result_list)) > (Leaguesize - 1) * (Leaguesize ) * 1/4:
                break
            # falldown
            sort_index2 = np.argsort(np.array([x.obj for x in teams_f.sol_list]))
            save_teams_num = sort_index2[0:(Leaguesize - NumberOfFall)]
            f_sol_list_copy = copy.deepcopy(teams_f.sol_list)
            save_teams = [f_sol_list_copy[i] for i in save_teams_num]
            teams_f.sol_list = []
            teams_f.sol_list = save_teams
            crossSol(teams_f)
            best_sol = copy.deepcopy(teams_f.best_sol.obj)
            history_best_obj.append(best_sol)
            upateTau(teams_f)  
            print('Best Cost of Iteration {} is {} '.format(i, best_sol))
        end_time = time.time() - start_time
        print("Finished epoch , took {} s".format(time.strftime('%H:%M:%S', time.gmtime(end_time))))    
        # plotObj(history_best_obj)
        # plotRoutes(teams_f)
        to_csv(local_result_list,store_result)
            
            
                    
                    
                    
                    
                
                
            
        
