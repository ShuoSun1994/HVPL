from turtle import pu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import copy
from MDVRPPDTW import Sol,run,calObj, plotObj, plotRoutes, outPut , Comparae_best_values,compute_and_comparae_best_values,movePosition,upateTau,crossSol
# 设置常量
sb = 0.2  # 队内学习最好球队概率
fallrate = 0.3
maxit = 10  # Number of iterations
Leaguesize = 10
# nPlayer = 10
# Leaguesize = 100
NumberOfFall = int(np.ceil(fallrate * Leaguesize))




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
    def __init__(self, teams_f, team1, team2, team1_index,Best_team ,sb,team2_index,rand_d_max = 0.4,rand_d_min = 0.1,worst_d_max = 4,worst_d_min = 4,regret_n = 5,r1 = 30,r2 = 18,r3 = 12,rho = 0.6,pu = 5,phi = 0.9):
        self.teams_f = teams_f
        self.team1 = copy.deepcopy(team1)
        self.team2 = copy.deepcopy(team2)
        self.index = team1_index
        self.y_index = team2_index
        self.Best_team = copy.deepcopy(Best_team)
        self.sb = sb
        self.rand_d_max = rand_d_max
        self.rand_d_min = rand_d_min
        self.worst_d_max = worst_d_max
        self.worst_d_min = worst_d_min
        self.regret_n = regret_n
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.rho = rho
        self.d_weight = np.ones(2) * 10
        self.d_select = np.zeros(2)
        self.d_score = np.zeros(2)
        self.d_history_select = np.zeros(2)
        self.d_history_score = np.zeros(2)
        self.r_weight = np.ones(3) * 10
        self.r_select = np.zeros(3)
        self.r_score = np.zeros(3)
        self.r_history_select = np.zeros(3)
        self.r_history_score = np.zeros(3)
        self.pu = pu
        self.phi = phi
        self.result = self.manage_strategy(regret_n)
        
        
    def manage_strategy(self,regret_n):
        # team1 
        self.strategy_x1(regret_n)   
        # team2    
        self.strategy_y1()
        if self.team1.obj > self.team2.obj:   
            self.strategy_x2()
        else:
            self.strategy_y2()
            
        return [self.team1,self.team2, self.Best_team]
    
    
    def resetScore(self):
    
        self.d_select = np.zeros(2)
        self.d_score = np.zeros(2)

        self.r_select = np.zeros(3)
        self.r_score = np.zeros(3)
   
    def selectDestoryRepair(self):
        d_weight=self.d_weight
        d_cumsumprob = (d_weight / sum(d_weight)).cumsum()  # cumsum 累积求和    array([0.5, 0.5]) -> array([0.5, 1. ]) 
        d_cumsumprob -= np.random.rand()
        destory_id= list(d_cumsumprob > 0).index(True)

        r_weight=self.r_weight
        r_cumsumprob = (r_weight / sum(r_weight)).cumsum()
        r_cumsumprob -= np.random.rand()
        repair_id = list(r_cumsumprob > 0).index(True)
        return destory_id,repair_id
   
    def createRandomDestory(self):
        d=random.uniform(self.rand_d_min,self.rand_d_max)
        reomve_list=random.sample(range(len(self.teams_f.demand_id_list)),int(d*len(self.teams_f.demand_id_list)))
        return reomve_list
    
    
    
    def createWorseDestory(self,sol):
        deta_f=[]
        for node_id in sol.node_id_list:
            sol_=copy.deepcopy(sol)
            sol_.node_id_list.remove(node_id)
            calObj(sol_,self.teams_f)
            deta_f.append(sol.obj-sol_.obj)
        sorted_id = sorted(range(len(deta_f)), key=lambda k: deta_f[k], reverse=True)
        d=random.randint(self.worst_d_min,self.worst_d_max)
        remove_list=sorted_id[:d]
        return remove_list
   
   
   
   
    def doDestory(self,destory_id,sol):
        if destory_id==0:
            reomve_list=self.createRandomDestory()
        else:
            reomve_list=self.createWorseDestory(sol)
        return reomve_list
    
    
    
    def createRandomRepair(self,remove_list,model,sol):
        unassigned_nodes_id=[]
        assigned_nodes_id = []
        # remove node from current solution
        for i in range(len(model.demand_id_list)):
            if i in remove_list:
                unassigned_nodes_id.append(sol.node_id_list[i])
            else:
                assigned_nodes_id.append(sol.node_id_list[i])
        # insert
        for node_id in unassigned_nodes_id:
            index=random.randint(0,len(assigned_nodes_id)-1)
            assigned_nodes_id.insert(index,node_id)
        new_sol=Sol()
        new_sol.node_id_list=copy.deepcopy(assigned_nodes_id)
        calObj(new_sol,model)
        return new_sol
    
    def findGreedyInsert(self,unassigned_nodes_id,assigned_nodes_id,model):
        best_insert_node_id=None
        best_insert_index = None
        best_insert_cost = float('inf')
        sol_1=Sol()
        sol_1.node_id_list=assigned_nodes_id
        calObj(sol_1,model)
        for node_id in unassigned_nodes_id:
            for i in range(len(assigned_nodes_id)):
                sol_2=Sol()
                sol_2.node_id_list=copy.deepcopy(assigned_nodes_id)
                sol_2.node_id_list.insert(i, node_id)
                calObj(sol_2, model)
                deta_f = sol_2.obj -sol_1.obj
                if deta_f<best_insert_cost:
                    best_insert_index=i
                    best_insert_node_id=node_id
                    best_insert_cost=deta_f
        return best_insert_node_id,best_insert_index
    
    
    
    
    def createGreedyRepair(self,remove_list,model,sol):
        unassigned_nodes_id = []
        assigned_nodes_id = []
        # remove node from current solution
        for i in range(len(model.demand_id_list)):
            if i in remove_list:
                unassigned_nodes_id.append(sol.node_id_list[i])
            else:
                assigned_nodes_id.append(sol.node_id_list[i])
        #insert
        while len(unassigned_nodes_id)>0:
            insert_node_id,insert_index=self.findGreedyInsert(unassigned_nodes_id,assigned_nodes_id,model)
            assigned_nodes_id.insert(insert_index,insert_node_id)
            unassigned_nodes_id.remove(insert_node_id)
        new_sol=Sol()
        new_sol.node_id_list=copy.deepcopy(assigned_nodes_id)
        calObj(new_sol,model)
        return new_sol
    
    
    def findRegretInsert(self,unassigned_nodes_id,assigned_nodes_id,model,regret_n):
        opt_insert_node_id = None
        opt_insert_index = None
        opt_insert_cost = -float('inf')
        sol_=Sol()
        for node_id in unassigned_nodes_id:
            n_insert_cost=np.zeros((len(assigned_nodes_id),3))
            for i in range(len(assigned_nodes_id)):
                sol_.node_id_list=copy.deepcopy(assigned_nodes_id)
                sol_.node_id_list.insert(i,node_id)
                calObj(sol_,model)
                n_insert_cost[i,0]=node_id
                n_insert_cost[i,1]=i
                n_insert_cost[i,2]=sol_.obj
            n_insert_cost= n_insert_cost[n_insert_cost[:, 2].argsort()]
            deta_f=0
            if self.regret_n > len(assigned_nodes_id):
                self.regret_n = random.randint(1,len(assigned_nodes_id))
            
            for i in range(1,self.regret_n):
                deta_f=deta_f+n_insert_cost[i,2]-n_insert_cost[0,2]
            self.regret_n = regret_n
            if deta_f>opt_insert_cost:
                opt_insert_node_id = int(n_insert_cost[0, 0])
                opt_insert_index=int(n_insert_cost[0,1])
                opt_insert_cost=deta_f
        return opt_insert_node_id,opt_insert_index
    
    
    
    def createRegretRepair(self,remove_list,model,sol,regret_n):
        unassigned_nodes_id = []
        assigned_nodes_id = []
        # remove node from current solution
        for i in range(len(model.demand_id_list)):
            if i in remove_list:
                unassigned_nodes_id.append(sol.node_id_list[i])
            else:
                assigned_nodes_id.append(sol.node_id_list[i])
        # insert
        while len(unassigned_nodes_id)>0:
            insert_node_id,insert_index=self.findRegretInsert(unassigned_nodes_id,assigned_nodes_id,model,regret_n)
            assigned_nodes_id.insert(insert_index,insert_node_id)
            unassigned_nodes_id.remove(insert_node_id)
        new_sol = Sol()
        new_sol.node_id_list = copy.deepcopy(assigned_nodes_id)
        calObj(new_sol, model)
        return new_sol 
    
    
    
    
    def doRepair(self,repair_id,reomve_list,model,sol,regret_n):
        if repair_id==0:
            new_sol=self.createRandomRepair(reomve_list,model,sol)
        elif repair_id==1:
            new_sol=self.createGreedyRepair(reomve_list,model,sol)
        else:
            new_sol=self.createRegretRepair(reomve_list,model,sol,regret_n)
        return new_sol
   
   
    def updateWeight(self):
    
        for i in range(self.d_weight.shape[0]):
            if self.d_select[i]>0:
                self.d_weight[i]=self.d_weight[i]*(1-self.rho)+self.rho*self.d_score[i]/self.d_select[i]
            else:
                self.d_weight[i] = self.d_weight[i] * (1 - self.rho)
        for i in range(self.r_weight.shape[0]):
            if self.r_select[i]>0:
                self.r_weight[i]=self.r_weight[i]*(1-self.rho)+self.rho*self.r_score[i]/self.r_select[i]
            else:
                self.r_weight[i] = self.r_weight[i] * (1 - self.rho)
        self.d_history_select = self.d_history_select + self.d_select
        self.d_history_score = self.d_history_score + self.d_score
        self.r_history_select = self.r_history_select + self.r_select
        self.r_history_score = self.r_history_score + self.r_score
   
   # ALNS
    def strategy_x1(self,regret_n):
        index = self.index
        sol = self.teams_f.sol_list[index]
        T = sol.obj * 0.2
        self.resetScore()
        for k in range(self.pu):
            destory_id,repair_id=self.selectDestoryRepair()
            self.d_select[destory_id]+=1
            self.r_select[repair_id]+=1
            reomve_list=self.doDestory(destory_id,sol)
            new_sol=self.doRepair(repair_id,reomve_list,self.teams_f,sol,regret_n)
            if new_sol.obj<sol.obj:
                sol=copy.deepcopy(new_sol)
                if new_sol.obj<self.teams_f.best_sol.obj:
                    self.teams_f.best_sol=copy.deepcopy(new_sol)
                    self.d_score[destory_id]+=self.r1
                    self.r_score[repair_id]+=self.r1
                else:
                    self.d_score[destory_id]+=self.r2
                    self.r_score[repair_id]+=self.r2
            elif new_sol.obj-sol.obj<T:
                sol=copy.deepcopy(new_sol)
                self.d_score[destory_id] += self.r3
                self.r_score[repair_id] += self.r3
            T=T*self.phi
        self.updateWeight()
            
            
             
    
    
     # 位置交换      
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
        
        
        

        # ACO
    def strategy_y1(self):
        index = self.y_index
        teams_f.sol_list[index] = None
        movePosition(self.teams_f,index)
        self.team2 = teams_f.sol_list[index]
        self.Best_team = teams_f.best_sol
    # 将最优解插入每个team
    def strategy_y2(self):
        sol_list = copy.deepcopy(self.teams_f.sol_list)
        best_team = copy.deepcopy(self.Best_team)
        index = self.y_index
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
def competition(A, B, teams_f, Best_team,sb):
    sol_list = copy.deepcopy(teams_f.sol_list)
    cost_list = [x.obj for x in sol_list]
    mincost = min(cost_list)
    maxcost = max(cost_list)
    m = np.zeros(len(teams_f.sol_list))
    for i in np.arange(len(teams_f.sol_list)):
        m[i] = (cost_list[i] - maxcost) / ((mincost - maxcost)+ 1*10**(-8) )

    MA = m[A] / np.sum(m)
    MB = m[B] / np.sum(m)
    x = sol_list[A]
    y = sol_list[B]
    d = MA / (MA + MB)
    r = np.random.rand()
    if d > r :
        result1 = ApplyStrategy(teams_f, x, y, A,Best_team,sb,B,rand_d_max = 0.1,rand_d_min = 0.1,worst_d_max = 20,worst_d_min = 5,regret_n = 5,r1 = 30,r2 = 18,r3 = 12,rho = 0.6,pu = 5,phi = 0.9).result
        X, Y ,new_Best_team = result1
        if X.obj < x.obj:
           x = X
           
        if Y.obj < y.obj:
               y = Y   
    else:
        result2 = ApplyStrategy(teams_f,y, x, B, Best_team,sb,A,rand_d_max = 0.1,rand_d_min = 0.1,worst_d_max = 20,worst_d_min = 5,regret_n = 5,r1 = 30,r2 = 18,r3 = 12,rho = 0.6,pu = 5,phi = 0.9).result
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

  #C1_20_n
    demand_file1 = './customers.csv'
    depot_file1 = './depots.csv'
    store_result1= './result.csv'
 
    file_tuple1 = (demand_file1,depot_file1,store_result1)
  
    file_list = [file_tuple1]
    # file_list = [file_tuple1,file_tuple2,file_tuple3,file_tuple4,file_tuple5,file_tuple6,file_tuple7,file_tuple8,file_tuple9,file_tuple10,file_tuple11,file_tuple12,file_tuple13,file_tuple14,file_tuple15,file_tuple16,file_tuple17,file_tuple18,file_tuple19,file_tuple20,file_tuple21,file_tuple22,file_tuple23,file_tuple24,file_tuple25,file_tuple26,file_tuple27,file_tuple28,file_tuple29,file_tuple30,file_tuple31,file_tuple32,file_tuple33,file_tuple34,file_tuple35,file_tuple36,file_tuple37,file_tuple38,file_tuple39,file_tuple40,file_tuple41,file_tuple42,file_tuple43,file_tuple44,file_tuple45,file_tuple46,file_tuple47,file_tuple48,file_tuple49,file_tuple50,file_tuple51,file_tuple52,file_tuple53,file_tuple54,file_tuple55,file_tuple56,file_tuple57,file_tuple58,file_tuple59,file_tuple60]
     
    for demand_file,depot_file,store_result in  file_list:
        start_time = time.time()
        teams_f = run(demand_file=demand_file,depot_file=depot_file,store_result = store_result,popsize=Leaguesize,v_cap=80,v_speed=1,opt_type=0,n_select=Leaguesize - NumberOfFall,pc=0.5,Q=10,tau0=10,alpha=1,beta=5,rho=0.1)
        best_team = Comparae_best_values(teams_f)
        Best_team = copy.deepcopy(best_team)
        history_best_obj = [] 
        # start loop
        for itr in np.arange(maxit):
            schedule = timetable(Leaguesize)
            local_result_list = []
            
            for i in np.arange(Leaguesize - 1):
                k = schedule[:,i]
                for j in np.arange(Leaguesize):
                    A = int(k[j])
                    B = j

                    x, y, Best_team_ = competition(A, B, teams_f, Best_team,sb)
                    Best_team = copy.deepcopy(Best_team_)
                    teams_f.sol_list[A] = x
                    teams_f.sol_list[B] = y
                  
                # learning phase 选出前三
                sort_index1 = np.argsort(np.array([x.obj for x in teams_f.sol_list]))
                rank1 = copy.deepcopy(teams_f.sol_list[sort_index1[0]])
                rank2 = copy.deepcopy(teams_f.sol_list[sort_index1[1]])
                rank3 = copy.deepcopy(teams_f.sol_list[sort_index1[2]])
                rank_dic = {i:x for i,x in enumerate([rank1, rank2,rank3])}
                sol_list_copy=copy.deepcopy(teams_f.sol_list)
                teams_f.sol_list=[] 
                for l in np.arange(Leaguesize):
                    rank_index = random.randint(0,len(rank_dic) - 1) #选择学习对象
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
                history_best_obj.append(teams_f.best_sol.obj)    
                if len(local_result_list) - len(set(local_result_list)) > (Leaguesize - 1) * (Leaguesize ) * 1/4:
                    break
                print('Best Cost of Week {} is {} '.format(i, teams_f.best_sol.obj)) 
                print('----------------')
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
            print('Best Cost of Iteration {} is {} '.format(itr, best_sol))
        end_time = time.time() - start_time
        print("Game Over , took {} s".format(time.strftime('%H:%M:%S', time.gmtime(end_time))))    
        # plotObj(history_best_obj)
        # plotRoutes(teams_f)
        to_csv(history_best_obj,store_result)
            
            
                    
                    
                    
                    
                
                
            
        
