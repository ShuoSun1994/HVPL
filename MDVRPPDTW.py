# -*- coding: utf-8 -*-
# @Time    : 2021/9/24 21:01
# @Author  : Praise
# @File    : GA_VRPTW.py
# obj:
import math
import random
import numpy as np
import copy
import xlsxwriter
import matplotlib.pyplot as plt
import csv
import sys
import pandas as pd

class  Sol():
    def __init__(self):
        self.obj=None   # 优化目标值
        self.node_id_list=[]  # 需求节点id有序排列集合
        self.cost_of_distance=None  # 距离成本
        self.cost_of_time=None  # 时间成本
        self.route_list=[]  # 车辆路径集合，对应MDVRPTW的解
        self.timetable_list=[]  # 车辆节点访问时间集合，对应MDVRPTW的解
        self.action_id=None
class Node():
    def __init__(self):
        self.id=0   #物理节点id，需唯一
        self.x_coord=0  # 物理节点x坐标
        self.y_cooord=0 # 物理节点y坐标
        self.demand=0 # 物理节点需求
        self.depot_capacity=0 # 车辆基地车队规模
        self.start_time=0 # 最早开始服务（被服务）时间
        self.end_time=1440 # 最晚结束服务（被服务）时间
        self.service_time=0  # 需求节点服务时间

class Model():
    def __init__(self):
        self.best_sol=None   # 全局最优解，值类型为Sol()
        self.demand_dict={}  # 需求节点集合（字典），值类型为Node()
        self.depot_dict={}   # 车场节点集合（字典），值类型为Node()
        self.depot_id_list=[] # 车场节点id集合
        self.demand_id_list=[] # 需求节点id集合
        self.sol_list=[] # 种群，值类型为Sol()
        self.distance_matrix={} # 节点距离矩阵
        self.time_matrix={} # 节点旅行时间矩阵
        self.number_of_demands=0 # 
        self.vehicle_cap=0
        self.vehicle_speed=1
        self.popsize=100 # 种群数量
        self.opt_type=1
        self.n_select=80
        self.alpha=2  # 
        self.beta=3
        self.Q=100
        self.tau0=10
        self.rho=0.5
        self.tau={}
        
        

# download datas
def readCSVFile(demand_file,depot_file,model):
    with open(demand_file,'r') as f:
        demand_reader=csv.DictReader(f)
        for row in demand_reader:
            node = Node()
            node.id = int(row['id'])
            node.x_coord = float(row['x_coord'])
            node.y_coord = float(row['y_coord'])
            node.demand = float(row['demand'])
            node.start_time=float(row['start_time'])
            node.end_time=float(row['end_time'])
            node.service_time=float(row['service_time'])
            model.demand_dict[node.id] = node
            model.demand_id_list.append(node.id)
        model.number_of_demands=len(model.demand_id_list)

    with open(depot_file, 'r') as f:
        depot_reader = csv.DictReader(f)
        for row in depot_reader:
            node = Node()
            node.id = row['id']
            node.x_coord = float(row['x_coord'])
            node.y_coord = float(row['y_coord'])
            node.depot_capacity = int(row['capacity'])   # vehcle numbers
            node.start_time=float(row['start_time'])
            node.end_time=float(row['end_time'])
            model.vehicle_cap = float(row['vehicle_capacity'])
            model.depot_dict[node.id] = node
            model.depot_id_list.append(node.id)

def calDistanceTimeMatrix(model):
    # 节点之间的距离矩阵
    for i in range(len(model.demand_id_list)):
        from_node_id = model.demand_id_list[i]
        for j in range(i + 1, len(model.demand_id_list)):
            to_node_id = model.demand_id_list[j]
            dist = math.sqrt((model.demand_dict[from_node_id].x_coord - model.demand_dict[to_node_id].x_coord) ** 2
                             + (model.demand_dict[from_node_id].y_coord - model.demand_dict[to_node_id].y_coord) ** 2)
            model.distance_matrix[from_node_id, to_node_id] = dist
            model.distance_matrix[to_node_id, from_node_id] = dist
            model.time_matrix[from_node_id,to_node_id] = math.ceil(dist/model.vehicle_speed)
            model.time_matrix[to_node_id,from_node_id] = math.ceil(dist/model.vehicle_speed)
            model.tau[from_node_id, to_node_id] = model.tau0
            model.tau[to_node_id, from_node_id] = model.tau0
        # 节点与配送中心之间的距离矩阵
        for _, depot in model.depot_dict.items():
            dist = math.sqrt((model.demand_dict[from_node_id].x_coord - depot.x_coord) ** 2
                             + (model.demand_dict[from_node_id].y_coord - depot.y_coord) ** 2)
            model.distance_matrix[from_node_id, depot.id] = dist
            model.distance_matrix[depot.id, from_node_id] = dist
            model.time_matrix[from_node_id,depot.id] = math.ceil(dist/model.vehicle_speed)
            model.time_matrix[depot.id,from_node_id] = math.ceil(dist/model.vehicle_speed)

def selectDepot(route,depot_dict,model):
    min_in_out_distance=float('inf')
    index=None
    for _,depot in depot_dict.items():
        if depot.depot_capacity>0:
            in_out_distance=model.distance_matrix[depot.id,route[0]]+model.distance_matrix[route[-1],depot.id]
            if in_out_distance<min_in_out_distance:
                index=depot.id
                min_in_out_distance=in_out_distance
    if index is None:
        print("there is no vehicle to dispatch")
        sys.exit(0)
    route.insert(0,index)
    route.append(index)
    depot_dict[index].depot_capacity=depot_dict[index].depot_capacity-1
    return route,depot_dict

def calTravelCost(route_list,model):
    timetable_list=[]
    cost_of_distance=0
    cost_of_time=0
    for route in route_list:
        timetable=[]
        for i in range(len(route)):
            if i == 0:
                depot_id=route[i]
                next_node_id=route[i+1]
                travel_time=model.time_matrix[depot_id,next_node_id]
                # 车辆始发的时间
                departure=max(0,model.demand_dict[next_node_id].start_time-travel_time)
                timetable.append((departure,departure))
            elif 1<= i <= len(route)-2:
                last_node_id=route[i-1]
                current_node_id=route[i]
                current_node = model.demand_dict[current_node_id]
                travel_time=model.time_matrix[last_node_id,current_node_id]
                arrival=max(timetable[-1][1]+travel_time,current_node.start_time)
                departure=arrival+current_node.service_time
                timetable.append((arrival,departure))
                cost_of_distance += model.distance_matrix[last_node_id, current_node_id]
                cost_of_time += model.time_matrix[last_node_id, current_node_id]+ current_node.service_time\
                                + max(current_node.start_time - timetable[-1][1] - travel_time, 0)  # 等待下一节点规定的开始时间
            else:
                last_node_id = route[i - 1]
                depot_id=route[i]
                travel_time = model.time_matrix[last_node_id,depot_id]
                departure = timetable[-1][1]+travel_time
                timetable.append((departure,departure))
                cost_of_distance +=model.distance_matrix[last_node_id,depot_id]
                cost_of_time+=model.time_matrix[last_node_id,depot_id]
        timetable_list.append(timetable)
    return timetable_list,cost_of_time,cost_of_distance

def extractRoutes(node_id_list,Pred,model):
    depot_dict=copy.deepcopy(model.depot_dict)
    route_list = []
    route = []
    label = Pred[node_id_list[0]]
    for node_id in node_id_list:
        if Pred[node_id] == label:
            route.append(node_id)
        else:
            route, depot_dict=selectDepot(route,depot_dict,model)
            route_list.append(route)
            route = [node_id]
            label = Pred[node_id]
    route, depot_dict = selectDepot(route, depot_dict, model)
    route_list.append(route)
    return route_list

def splitRoutes(node_id_list,model):
    depot=model.depot_id_list[0]
    V={id:float('inf') for id in model.demand_id_list}  # 节点标签
    V[depot]=0   # 车从配送站出发时空载
    Pred={id:depot for id in model.demand_id_list}  # 前向节点id
    for i in range(len(node_id_list)):
        n_1=node_id_list[i]   #  起始客户点
        # demand=0
        demand = model.vehicle_cap
        departure=0
        j=i
        cost=0
        while True:
            n_2 = node_id_list[j]
            # demand = demand + model.demand_dict[n_2].demand
            demand = demand - model.demand_dict[n_2].demand
            # 当客户节点是车辆第一个到达的时
            if n_1 == n_2:  
                # 如果车辆到达时间早于客户要求的到达时间，车辆可以等到规定时间再服务客户,要以客户要求时间为准;
                # 如果车辆到达时间晚于客户要求的到达时间，则只能按车辆到达时间为准
                arrival= max(model.demand_dict[n_2].start_time,model.depot_dict[depot].start_time+model.time_matrix[depot,n_2])
                departure=arrival+model.demand_dict[n_2].service_time
                if model.opt_type == 0:
                    cost=model.distance_matrix[depot,n_2]*2
                else:
                    cost=model.time_matrix[depot,n_2]*2
            else:
                n_3=node_id_list[j-1]  # 上一个节点
                arrival = max(departure + model.time_matrix[n_3, n_2], model.demand_dict[n_2].start_time)
                departure = arrival + model.demand_dict[n_2].service_time
                if model.opt_type == 0:
                    # 一直在求环路距离，即从配送站出发回到配送站的距离
                    cost=cost-model.distance_matrix[n_3,depot]+model.distance_matrix[n_3,n_2]+model.distance_matrix[n_2,depot]
                else:
                    cost = cost - model.time_matrix[n_3, depot] + model.time_matrix[n_3, n_2] \
                           + max(model.demand_dict[n_2].start_time - arrival, 0) + model.time_matrix[n_2, depot]
            # if demand<=model.vehicle_cap and departure <= model.demand_dict[n_2].end_time:
            if 0<=demand<=model.vehicle_cap and departure <= model.demand_dict[n_2].end_time:
                if departure+model.time_matrix[n_2,depot] <= model.depot_dict[depot].end_time:
                    n_4=node_id_list[i-1] if i-1>=0 else depot  # juge
                    if V[n_4]+cost <= V[n_2]:
                        V[n_2]=V[n_4]+cost
                        Pred[n_2]=i-1
                    j=j+1
            else:
                break
            if j==len(node_id_list):
                break
    route_list= extractRoutes(node_id_list,Pred,model)
    return len(route_list),route_list

def Comparae_best_values(model):
    best_sol=Sol()
    best_sol.obj=float('inf')

    # calculate travel distance and travel time
    for sol in model.sol_list:
        node_id_list=copy.deepcopy(sol.node_id_list)
        num_vehicle, route_list = splitRoutes(node_id_list, model)
        # travel cost
        timetable_list,cost_of_time,cost_of_distance =calTravelCost(route_list,model)
        if model.opt_type == 0:
            sol.obj=cost_of_distance
        else:
            sol.obj=cost_of_time
        sol.route_list = route_list
        sol.timetable_list = timetable_list
        sol.cost_of_distance=cost_of_distance
        sol.cost_of_time=cost_of_time
        if sol.obj < best_sol.obj:
            best_sol=copy.deepcopy(sol)
    if best_sol.obj<model.best_sol.obj:
        model.best_sol=copy.deepcopy(best_sol)
    return model.best_sol



def generateInitialSol(model):
    demand_id_list=copy.deepcopy(model.demand_id_list)
    for i in range(model.popsize):
        seed=int(random.randint(0,10))
        random.seed(seed)
        random.shuffle(demand_id_list)
        sol=Sol()
        sol.node_id_list=copy.deepcopy(demand_id_list)
        model.sol_list.append(sol)


def compute_and_comparae_best_values(model,best_sol,sol):
    # calculate travel distance and travel time
    node_id_list=copy.deepcopy(sol.node_id_list)
    num_vehicle, route_list = splitRoutes(node_id_list, model)
    # travel cost
    timetable_list,cost_of_time,cost_of_distance =calTravelCost(route_list,model)
    if model.opt_type == 0:
        sol.obj=cost_of_distance
    else:
        sol.obj=cost_of_time
    sol.route_list = route_list
    sol.timetable_list = timetable_list
    sol.cost_of_distance=cost_of_distance
    sol.cost_of_time=cost_of_time
    if sol.obj < best_sol.obj:
        best_sol=copy.deepcopy(sol)
    if best_sol.obj<model.best_sol.obj:
        model.best_sol=copy.deepcopy(best_sol)
  

def selectSol(model):
    sol_list=copy.deepcopy(model.sol_list)
    model.sol_list=[]
    for i in range(model.n_select):
        f1_index=random.randint(0,len(sol_list)-1)
        f2_index=random.randint(0,len(sol_list)-1)
        f1_fit=sol_list[f1_index].fitness
        f2_fit=sol_list[f2_index].fitness
        if f1_fit<f2_fit:
            model.sol_list.append(sol_list[f2_index])
        else:
            model.sol_list.append(sol_list[f1_index])


def calObj(sol,model):

    node_id_list=copy.deepcopy(sol.node_id_list)
    num_vehicle, sol.route_list = splitRoutes(node_id_list, model)
    # travel cost
    sol.timetable_list,sol.cost_of_time,sol.cost_of_distance =calTravelCost(sol.route_list,model)
    if model.opt_type == 0:
        sol.obj=sol.cost_of_distance
    else:
        sol.obj=sol.cost_of_time

def movePosition(model,index):
    nodes_id=[int(random.randint(0,len(model.demand_id_list)-1))]
    all_nodes_id=copy.deepcopy(model.demand_id_list)
    all_nodes_id.remove(nodes_id[-1])
    while len(all_nodes_id)>0:
        next_node_no=searchNextNode(model,nodes_id[-1],all_nodes_id)
        nodes_id.append(next_node_no)
        all_nodes_id.remove(next_node_no)
    sol=Sol()
    sol.node_id_list=nodes_id
    calObj(sol,model)
    model.sol_list[index]=copy.deepcopy(sol)
    if sol.obj<model.best_sol.obj:
        model.best_sol=copy.deepcopy(sol)

def crossSol(model):
    sol_list=copy.deepcopy(model.sol_list)
    while True:
        f1_index = random.randint(0, len(sol_list) - 1)
        f2_index = random.randint(0, len(sol_list) - 1)
        if f1_index!=f2_index:
            # 挑选两个解序列
            f1 = copy.deepcopy(sol_list[f1_index])  
            f2 = copy.deepcopy(sol_list[f2_index])
            if random.random() <= model.pc:
                # 挑选解序列中某个位置
                cro1_index=int(random.randint(0,len(model.demand_id_list)-1))
                cro2_index=int(random.randint(cro1_index,len(model.demand_id_list)-1))
                new_c1_f = []  # 新的解序列前段
                new_c1_m=f1.node_id_list[cro1_index:cro2_index+1]  # 新的解序列中段
                new_c1_b = []    # 新的解序列后段
                new_c2_f = []
                new_c2_m=f2.node_id_list[cro1_index:cro2_index+1]
                new_c2_b = []
                for index in range(len(model.demand_id_list)):
                    # 之所以分前中后 是因为 这样相当于将另外一个解序列的一段拼接上去
                    if len(new_c1_f)<cro1_index:  
                        if f2.node_id_list[index] not in new_c1_m:
                            new_c1_f.append(f2.node_id_list[index])
                    else:
                        if f2.node_id_list[index] not in new_c1_m:
                            new_c1_b.append(f2.node_id_list[index])
                for index in range(len(model.demand_id_list)):
                    if len(new_c2_f)<cro1_index:
                        if f1.node_id_list[index] not in new_c2_m:
                            new_c2_f.append(f1.node_id_list[index])
                    else:
                        if f1.node_id_list[index] not in new_c2_m:
                            new_c2_b.append(f1.node_id_list[index])
                new_c1=copy.deepcopy(new_c1_f)
                new_c1.extend(new_c1_m)
                new_c1.extend(new_c1_b)
                f1.nodes_seq=new_c1
                best_sol =copy.deepcopy( model.best_sol)
                compute_and_comparae_best_values(model,best_sol,f1)
                new_c2=copy.deepcopy(new_c2_f)
                new_c2.extend(new_c2_m)
                new_c2.extend(new_c2_b)
                f2.nodes_seq=new_c2
                best_sol =copy.deepcopy( model.best_sol)
                compute_and_comparae_best_values(model,best_sol,f2)
                model.sol_list.append(copy.deepcopy(f1))
                model.sol_list.append(copy.deepcopy(f2))
            else:
                model.sol_list.append(copy.deepcopy(f1))
                model.sol_list.append(copy.deepcopy(f2))
            if len(model.sol_list)>= model.popsize:
                break



def searchNextNode(model,current_node_id,SE_List):
    prob=np.zeros(len(SE_List))
    for i,node_id in enumerate(SE_List):
        eta=1/model.distance_matrix[current_node_id,node_id]
        tau=model.tau[current_node_id,node_id]
        prob[i]=((eta**model.alpha)*(tau**model.beta))
    cumsumprob=(prob/sum(prob)).cumsum()
    cumsumprob -= np.random.rand()
    next_node_id= SE_List[list(cumsumprob > 0).index(True)]
    return next_node_id

def upateTau(model):
    rho=model.rho
    for k in model.tau.keys():
        model.tau[k]=(1-rho)*model.tau[k]
    for sol in model.sol_list:
        nodes_id=sol.node_id_list
        for i in range(len(nodes_id)-1):
            from_node_id=nodes_id[i]
            to_node_id=nodes_id[i+1]
            model.tau[from_node_id,to_node_id]+=model.Q/sol.obj



def plotObj(obj_list):
    plt.rcParams['font.sans-serif'] = ['SimHei'] #show chinese
    plt.rcParams['axes.unicode_minus'] = False  # Show minus sign
    plt.plot(np.arange(1,len(obj_list)+1),obj_list)
    plt.xlabel('Iterations')
    plt.ylabel('Obj Value')
    plt.grid()
    plt.xlim(1,len(obj_list)+1)
    plt.show()

def outPut(model):
    work=xlsxwriter.Workbook('result.xlsx')
    worksheet=work.add_worksheet()
    worksheet.write(0, 0, 'cost_of_time')
    worksheet.write(0, 1, 'cost_of_distance')
    worksheet.write(0, 2, 'opt_type')
    worksheet.write(0, 3, 'obj')
    worksheet.write(1,0,model.best_sol.cost_of_time)
    worksheet.write(1,1,model.best_sol.cost_of_distance)
    worksheet.write(1,2,model.opt_type)
    worksheet.write(1,3,model.best_sol.obj)
    worksheet.write(2,0,'vehicleID')
    worksheet.write(2,1,'route')
    worksheet.write(2,2,'timetable')
    for row,route in enumerate(model.best_sol.route_list):
        worksheet.write(row+3,0,'v'+str(row+1))
        r=[str(i)for i in route]
        worksheet.write(row+3,1, '-'.join(r))
        r=[str(i)for i in model.best_sol.timetable_list[row]]
        worksheet.write(row+3,2, '-'.join(r))
    work.close()

def plotRoutes(model):
    for route in model.best_sol.route_list:
        x_coord=[model.depot_dict[route[0]].x_coord]
        y_coord=[model.depot_dict[route[0]].y_coord]
        for node_id in route[1:-1]:
            x_coord.append(model.demand_dict[node_id].x_coord)
            y_coord.append(model.demand_dict[node_id].y_coord)
        x_coord.append(model.depot_dict[route[-1]].x_coord)
        y_coord.append(model.depot_dict[route[-1]].y_coord)
        plt.grid()
        if route[0]=='d1':
            plt.plot(x_coord,y_coord,marker='o',color='black',linewidth=0.5,markersize=5)
        elif route[0]=='d2':
            plt.plot(x_coord,y_coord,marker='o',color='orange',linewidth=0.5,markersize=5)
        else:
            plt.plot(x_coord,y_coord,marker='o',color='b',linewidth=0.5,markersize=5)
    plt.xlabel('x_coord')
    plt.ylabel('y_coord')
    plt.show()



def run(demand_file,depot_file,store_result,popsize,v_cap,v_speed,opt_type,n_select,pc=0.5,Q=10,tau0=10,alpha=1,beta=5,rho=0.1):
    """
    :param demand_file: demand file path
    :param depot_file: depot file path
    :param popsize: Population size
    :param v_cap: Vehicle capacity
    :param v_speed: Vehicle free speed
    :param opt_type: Optimization type:0:Minimize the cost of travel distance;1:Minimize the cost of travel time
    :return:
    """
    model=Model()
    model.vehicle_cap=v_cap
    model.vehicle_speed = v_speed
    model.popsize=popsize
    model.opt_type=opt_type
    model.n_select=n_select
    model.pc=0.5
    model.alpha=alpha
    model.beta=beta
    model.Q=Q
    model.tau0=tau0
    model.rho=rho
    readCSVFile(demand_file,depot_file,model)
    calDistanceTimeMatrix(model)
    generateInitialSol(model)
    best_sol=Sol()
    best_sol.obj=float('inf')
    model.best_sol=best_sol
    return model
   

if __name__=='__main__':
    # popsize - 种群规模
    # v_cap - 车辆容量
    # v_speed - 车辆行驶速度，用于计算旅行时间
    # opt_type - 0：最小旅行距离，1：最小时间成本
    demand_file='/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPTW/demand.csv'
    depot_file='/home/ss/Desktop/科研/code/code/paper_code/VPL/data/MDVRPTW/depot.csv'
    model = run(demand_file=demand_file,depot_file=depot_file,store_result =store_result,popsize=100,
       v_cap=80,v_speed=1,opt_type=0,n_select=80,Q=10,tau0=10,alpha=1,beta=5,rho=0.1)
