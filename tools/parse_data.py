import pickle, sys
import argparse
import os
import time
import numpy as np
import random
import copy
from scipy.spatial import KDTree
sys.path.append('../')
from simulator.utilities import *

class kddcup2020:
    def __init__(self,args=None) :
        self.data_dir='../data_for_simulator/KDDcup2020/'
        r=6371
        per_lon=r/360*2*3.14159*np.cos(np.pi/6)
        per_lat=r/360*2*3.14159
        self.per_lonlat=np.array([per_lon,per_lat])
        self.weekday=np.array([1,2,3,4,7,8,9,10,11,14,15,16,17,18,21,22,23,24,25,28,29,30])-1
        self.weekend=np.array([5,6,12,13,19,20,26,27])-1
        self.speed=args.speed    #min/km
        self.driver_num=args.driver_num
        self.wait_time=args.wait_time
        self.grid_num=8518
        self.row=None
        self.col=None

    def cal_dis(self,array1,array2):
        return np.sqrt(np.sum((np.abs(array1-array2)*self.per_lonlat)**2))

    def load_grid(self):
        map_data=np.loadtxt(self.data_dir+"hexagon_grid_table.csv",dtype=str)
        map_xy=[]
        map_id=[]
        for i,grid in enumerate(map_data):
            if i!=4183:
                grid=grid.split(',')
                map_id.append(grid[0])
                map_xy.append(np.reshape(np.array(grid[1:],dtype=float),[6,2])[None,:,:])
        map_xy=np.concatenate(map_xy,axis=0)
        grid_mean=np.mean(map_xy,axis=1)    
        assert self.grid_num==len(grid_mean)
        assert grid_mean.shape==(8518,2) 
        return grid_mean

    def load_grid_neighbor(self):
        with open(self.data_dir+'processed/grid_neighbor.pkl', 'rb') as handle:
            grid_neighbor = pickle.load(handle) 
        # grid_neighbor[i]=(dis,neighbor)
        grid_neighbor_process=[]
        for id in range(self.grid_num):
            neighbor={}
            for r in range(5):
                neighbor[r]=[]
            dis=np.round(grid_neighbor[id][0]).astype(int)
            for i in range(len(dis)):
                if dis[i] <5:
                    neighbor[dis[i]].append(grid_neighbor[id][1][i])
            grid_neighbor_process.append(neighbor)
        # grid_neighbor_process[i]={r:[id]}
        return grid_neighbor_process

    def load_map500(self):
        # dis每1.3公里是一个neighbor
        with open(self.data_dir+'processed/map_500_neighbor.pkl', 'rb') as handle:
            grid_neighbor = pickle.load(handle) 
        # grid_neighbor[i]=(dis,neighbor)
        grid_neighbor_process=[]
        for id in range(self.grid_num):
            neighbor={}
            for r in range(5):
                neighbor[r]=[]
            dis=np.round(grid_neighbor[id][0]).astype(int)
            for i in range(len(dis)):
                if dis[i] <5:
                    neighbor[dis[i]].append(grid_neighbor[id][1][i])
            grid_neighbor_process.append(neighbor)
        # grid_neighbor_process[i]={r:[id]}
        return grid_neighbor_process

    def cal_dis_lonlat(self,array1,array2):
        return np.sqrt(np.sum((np.abs(array1-array2)*self.per_lonlat)**2))

    def load_order(self):
        with open(self.data_dir+'processed/order_processed.pkl', 'rb') as handle:
            data = pickle.load(handle)
        start_grid,end_grid,start_time,duration,price=[],[],[],[],[]
        for day in self.weekday:
            start_grid+=data[day]['start_grid']
            end_grid+=data[day]['end_grid']
            price+=data[day]['price']
            for i in range(len(data[day]['price'])):
                start_t=data[day]['start_time'][i]
                end_t=data[day]['end_time'][i]
                duration.append(int((end_t-start_t)/60))
                t=time.localtime(start_t)
                start_time.append(t.tm_hour*60+t.tm_min)
        start_grid=np.array(start_grid,dtype=int)
        end_grid=np.array(end_grid,dtype=int)
        start_time=np.array(start_time,dtype=int)
        duration=np.array(duration,dtype=int)
        price=np.array(price,dtype=float)
        index= ~((duration<3)+(duration>=180))
        order={
            'start_grid': start_grid[index],
            'end_grid': end_grid[index],
            'start_time': start_time[index],
            'duration':duration[index],
            'price':price[index]
        }
        self.order_ratio=1/len(self.weekday)

        # 统计每个网格的order数量，作为driver的初始化的参考
        order_grid=np.zeros(self.grid_num,dtype=int)
        for id in start_grid:
            order_grid[id]+=1
        logit=np.log(order_grid+2)
        driver_locate_p=logit/np.sum(logit)
        return order, driver_locate_p

    def partition_map_500(self):
        # partitions process 8518 grids into 500 grids 
        # in the way of matrix
        corner=[
            [103.9221,30.7909],
            [103.9221,30.5592],
            [104.2117,30.7909],
            [104.2117,30.5592]
        ]
        corner=np.array(corner)
        radius=self.cal_dis(corner[0],corner[1])/20
        lat_num=int(self.cal_dis(corner[0],corner[1])/radius)
        lon_num=int(round(self.cal_dis(corner[0],corner[2])/(radius*np.sqrt(3)/2)))
        lat_per_radius=(corner[0,1]-corner[1,1])/lat_num
        lon_per_radius=(corner[2,0]-corner[0,0])/lon_num
        assert lat_num==20 and lon_num==25 , print(lat_num,lon_num)
        grid_small=np.zeros((lat_num,lon_num,2))
        for i in range(lat_num):
            for j in range(lon_num):
                grid_small[i,j][0]=lon_per_radius*j+corner[0,0]
                if j%2==0:
                    grid_small[i,j][1]=-lat_per_radius*i+corner[0,1]
                else:
                    grid_small[i,j][1]=-lat_per_radius*(i+0.5)+corner[0,1]
        with open('../data_for_simulator/KDDcup2020/processed/map_500.pkl', 'wb') as handle:
            pickle.dump(grid_small, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # process neighbor  of map
        grid_small=np.reshape(grid_small,(-1,2))
        kdtree=KDTree(list(grid_small*self.per_lonlat))
        grid_neighbor={}
        for i in range(len(grid_small)):
            dis,neighbor= kdtree.query(grid_small[i]*self.per_lonlat,40)
            grid_neighbor[i]=(dis,neighbor)
        # dis每1.3公里是一个neighbor
        with open('../data_for_simulator/KDDcup2020/processed/map_500_neighbor.pkl', 'wb') as handle:
            pickle.dump(grid_neighbor, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def process_order(self):
        # process order
        order=[]
        for day in range(1,31):
            with open('total_ride_request/order_201611'+format(day,'0>2'), 'r') as f:
                data=f.readlines()
            order_id,start_time, end_time, start_grid, end_grid, price=[],[],[],[],[],[]
            for i,row in enumerate(data):
                # '39d471630d26439111a7746d164d34eb,1478091677,1478092890,104.00816,30.70622,104.064147,30.685848,3.82\n'
                row=row.split(',')
                order_id.append(row[0])
                start_time.append(int(row[1]))
                end_time.append(int(row[2]))
                dis,id=kdtree.query([float(row[3]),float(row[4])])
                start_grid.append(id)
                dis,id=kdtree.query([float(row[5]),float(row[6])])
                end_grid.append(id)
                price.append(float(row[7]))
            order.append({
                'data': day,
                'start_time':start_time,
                'end_time':end_time,
                'start_grid':start_grid,
                'end_grid':end_grid,
                'price':price
            })
        
    def process_grid_neighbor(self):
        grid_mean=self.load_grid()
        # 计算每个grid与其它grid的距离
        kdtree=KDTree(list(grid_mean*self.per_lonlat))
        grid_neighbor={}
        for i in range(len(grid_mean)):
            dis,neighbor= kdtree.query(grid_mean[i]*self.per_lonlat,100)
            grid_neighbor[i]=(dis,neighbor)
        with open('../data_for_simulator/KDDcup2020/processed/grid_neighbor.pkl', 'wb') as handle:
            pickle.dump(grid_neighbor, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def visualize(self):
        with open(self.data_dir+'processed/order_processed.pkl', 'rb') as handle:
            data = pickle.load(handle)
        day=0
        order_hour=np.zeros(24,dtype=int)
        for i in range(len(order_data[day]['start_grid'])):
            start_t=data[day]['start_time'][i]
            hour=time.localtime(start_t).tm_hour
            order_grid_hour[hour,order_data[day]['start_grid'][i]]+=1

class map_regular:
    def __init__(self,args=None) :
        self.data_dir='../data_for_simulator/KDDcup2020/'
        r=6371
        per_lon=r/360*2*3.14159*np.cos(np.pi/6)
        per_lat=r/360*2*3.14159
        self.per_lonlat=np.array([per_lon,per_lat])
        self.weekday=np.array([1,2,3,4,7,8,9,10,11,14,15,16,17,18,21,22,23,24,25,28,29,30])-1
        self.weekend=np.array([5,6,12,13,19,20,26,27])-1
        self.speed=args.speed    #min/km
        self.driver_num=args.driver_num
        self.wait_time=args.wait_time
        self.grid_num=args.grid_num

    def cal_dis(self,array1,array2):
        return np.sqrt(np.sum((np.abs(array1-array2)*self.per_lonlat)**2))

    def load_grid(self):
        with open(self.data_dir+'processed/map_{}.pkl'.format(self.grid_num), 'rb') as handle:
            grid= pickle.load(handle) 
        self.row=grid.shape[0]
        self.col=grid.shape[1]
        grid=np.reshape(grid,(-1,2))
        return grid

    def load_grid_neighbor(self):
        with open(self.data_dir+'processed/map_{}_neighbor.pkl'.format(self.grid_num), 'rb') as handle:
            grid_neighbor = pickle.load(handle) 
        '''
        # grid_neighbor[i]=(dis,neighbor)
        # dis每1.3公里是一个neighbor
        dis_per_neighbor=1.2
        grid_neighbor_process=[]
        for id in range(self.grid_num):
            neighbor={}
            for r in range(4):
                neighbor[r]=[]
            dis=np.round(grid_neighbor[id][0]/dis_per_neighbor).astype(int)
            for i in range(len(dis)):
                if dis[i] <4:
                    neighbor[dis[i]].append(grid_neighbor[id][1][i])
            grid_neighbor_process.append(neighbor)
         '''
        grid_neighbor_process=[]
        for id,grid in grid_neighbor.items():
            neighbor={}
            neighbor[0]=[grid[0]]
            neighbor[1]=grid[1:7]
            neighbor[2]=grid[7:]
            grid_neighbor_process.append(neighbor)
        # grid_neighbor_process[i]={r:[id]}
        return grid_neighbor_process

    def load_order(self,distri):
        with open(self.data_dir+'processed/map_{}_order.pkl'.format(self.grid_num), 'rb') as handle:
            data = pickle.load(handle)
        start_grid,end_grid,start_time,duration,price=[],[],[],[],[]
        for day in self.weekday:
            start_grid+=data[day]['start_grid']
            end_grid+=data[day]['end_grid']
            price+=data[day]['price']
            for i in range(len(data[day]['price'])):
                start_t=data[day]['start_time'][i]
                end_t=data[day]['end_time'][i]
                duration.append(int((end_t-start_t)/60))
                t=time.localtime(start_t)
                start_time.append(t.tm_hour*60+t.tm_min)
        start_grid=np.array(start_grid,dtype=int)
        end_grid=np.array(end_grid,dtype=int)
        start_time=np.array(start_time,dtype=int)
        duration=np.array(duration,dtype=int)
        price=np.array(price,dtype=float)
        index= ~((duration<3)+(duration>=180))
        if distri is not None:
            pro=np.random.rand(len(index))
            for i,id in enumerate(start_grid):
                if pro[i]>distri[id]:
                    index[i]=False

        order={
            'start_grid': start_grid[index],
            'end_grid': end_grid[index],
            'start_time': start_time[index],
            'duration':duration[index],
            'price':price[index]
        }
        self.order_ratio=1/len(self.weekday)

        # 统计每个网格的order数量，作为driver的初始化的参考
        order_grid=np.zeros(self.grid_num,dtype=int)
        order_des_grid=np.zeros(self.grid_num,dtype=int)
        for id in start_grid:
            order_grid[id]+=1
        for id in end_grid:
            order_des_grid[id]+=1
        logit=np.log(order_grid+2)
        driver_locate_p=logit/np.sum(logit)
        return order, driver_locate_p

    def process_grid(self):
        # partitions process 8518 grids into args.grdi_num grids 
        if self.grid_num==500:
            corner=[
            [103.9221,30.7909],
            [103.9221,30.5592],
            [104.2117,30.7909],
            [104.2117,30.5592]]
            corner=np.array(corner)
            radius=self.cal_dis(corner[0],corner[1])/20
        elif self.grid_num==130:
            corner=[
            [103.9221,30.7909],
            [103.9221,30.5592],
            [104.2234,30.7909],
            [104.2234,30.5592]]
            corner=np.array(corner)
            radius=self.cal_dis(corner[0],corner[1])/10
        elif self.grid_num==35:
            corner=[
            [103.9221,30.7909],
            [103.9221,30.5592],
            [104.2466,30.7909],
            [104.2466,30.5592]]
            corner=np.array(corner)
            radius=self.cal_dis(corner[0],corner[1])/5
        elif self.grid_num==30:
            corner=[
            [104.0000,30.7200],
            [104.0000,30.6200],
            [104.1400,30.7200],
            [104.1400,30.6200]]
            corner=np.array(corner)
            radius=self.cal_dis(corner[0],corner[1])/5
        lat_num=int(self.cal_dis(corner[0],corner[1])/radius)
        lon_num=int(round(self.cal_dis(corner[0],corner[2])/(radius*np.sqrt(3)/2)))
        lat_per_radius=(corner[0,1]-corner[1,1])/lat_num
        lon_per_radius=(corner[2,0]-corner[0,0])/lon_num
        grid_small=np.zeros((lat_num,lon_num,2))
        for i in range(lat_num):
            for j in range(lon_num):
                grid_small[i,j][0]=lon_per_radius*j+corner[0,0]
                if j%2==0:
                    grid_small[i,j][1]=-lat_per_radius*i+corner[0,1]
                else:
                    grid_small[i,j][1]=-lat_per_radius*(i+0.5)+corner[0,1]
        with open('../data_for_simulator/KDDcup2020/processed/map_{}.pkl'.format(self.grid_num), 'wb') as handle:
            pickle.dump(grid_small, handle, protocol=pickle.HIGHEST_PROTOCOL)
       
    def process_grid_neighbor(self):
        # process neighbor  of map
        '''
        grid_small=np.reshape(grid_small,(-1,2))
        kdtree=KDTree(list(grid_small*self.per_lonlat))
        grid_neighbor={}
        for i in range(len(grid_small)):
            dis,neighbor= kdtree.query(grid_small[i]*self.per_lonlat,40)
            grid_neighbor[i]=(dis,neighbor)
        # dis每1.3公里是一个neighbor
        with open('../data_for_simulator/KDDcup2020/processed/map_500_neighbor.pkl', 'wb') as handle:
            pickle.dump(grid_neighbor, handle, protocol=pickle.HIGHEST_PROTOCOL)
         '''
        # process neighbor
        grid_mean=self.load_grid()
        M=self.row
        N=self.col
        neighbor_grid={}
        for i,id in enumerate(range(len(grid_mean))):
            neighbor=[id]
            neighbor+=get_neighbor_list(id,1,M,N)
            neighbor+=get_neighbor_list(id,2,M,N)
            neighbor_grid[i]=neighbor
        with open('../data_for_simulator/KDDcup2020/processed/map_{}_neighbor.pkl'.format(self.grid_num), 'wb') as handle:
                pickle.dump(neighbor_grid, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def process_order(self):
        grid=self.load_grid()
        kdtree=KDTree(list(grid*self.per_lonlat))
        # process order
        order=[]
        for day in range(1,31):
            with open(self.data_dir+'total_ride_request/order_201611'+format(day,'0>2'), 'r') as f:
                data=f.readlines()
            order_id,start_time, end_time, start_grid, end_grid, price=[],[],[],[],[],[]
            for i,row in enumerate(data):
                # '39d471630d26439111a7746d164d34eb,1478091677,1478092890,104.00816,30.70622,104.064147,30.685848,3.82\n'
                row=row.split(',')
                dis_start,id_start=kdtree.query(np.array([float(row[3]),float(row[4])])*self.per_lonlat)
                dis_end,id_end=kdtree.query(np.array([float(row[5]),float(row[6])])*self.per_lonlat)
                if dis_start<=2.6 and dis_end<=2.6:
                    start_grid.append(id_start)
                    end_grid.append(id_end)
                    order_id.append(row[0])
                    start_time.append(int(row[1]))
                    end_time.append(int(row[2]))
                    price.append(float(row[7]))
            order.append({
                'data': day,
                'start_time':start_time,
                'end_time':end_time,
                'start_grid':start_grid,
                'end_grid':end_grid,
                'price':price
            })
        with open('../data_for_simulator/KDDcup2020/processed/map_{}_order.pkl'.format(self.grid_num), 'wb') as handle:
            pickle.dump(order, handle, protocol=pickle.HIGHEST_PROTOCOL)

class kdd18:
    def __init__(self,args=None) :
        self.data_dir='../data_for_simulator/KDDcup2020/'
        #self.speed=args.speed    #min/km
        self.driver_num=args.driver_num
        self.wait_time=args.wait_time
        self.TIME_LEN=args.TIME_LEN
        self.dispatch_interval=args.dispatch_interval
        self.grid_num=args.grid_num
        try :
            self.change_order_distri=args.change_order_distri
        except:
            self.change_order_distri=False
        np.random.seed(0)
        random.seed(0)

    def change_distri(self):
        center=int(np.round(self.grid_num/2))
        center_xy=ids_1dto2d(center,self.M,self.N)
        distri=np.zeros(self.grid_num)
        for i in range(self.grid_num):
            cur_xy=ids_1dto2d(i,self.M,self.N)
            distri[i]= np.sqrt((center_xy[0]-cur_xy[0])**2+(center_xy[1]-cur_xy[1])**2)
        distri[center]=1
        distri= 1/distri
        return distri

    def build_dataset(self,args):
        self.order_time_dist = []        # no use
        self.order_price_dist = []       # no use
        self.order_num_dist=[]           # no use
        self.onoff_driver_location_mat=[]   # 不考虑车辆的随机上线离线
        self.idle_driver_dist_time=np.zeros((self.TIME_LEN,2))
        self.idle_driver_dist_time[:,0]=self.driver_num     # 仅用于初始化汽车数量
        self.target_ids=[1]*self.grid_num
        self.dataset=map_regular(args)
        self.grid_xy=self.dataset.load_grid()
        self.M=self.dataset.row
        self.N=self.dataset.col

        if self.change_order_distri:
            self.distri=self.change_distri()
        else:
            self.distri=None

        self.mapped_matrix_int= np.reshape(np.arange(self.grid_num),(self.M,self.N))
        self.grid_neighbor=self.dataset.load_grid_neighbor()
        self.idle_driver_location_mat=np.zeros((self.TIME_LEN,self.grid_num))   # 仅用于初始化汽车分布
        #self.order_real  # N*[ori id,end id, start time, duration,price]
        order,driver_locate_p=self.dataset.load_order(self.distri)
        if self.grid_num==35:
            #driver_locate_p[:25]=0.0002
            #driver_locate_p/=np.sum(driver_locate_p)
            pass
        self.order_ratio=self.dataset.order_ratio
        start_grid=order['start_grid'][:,None].astype(np.float)
        end_grid=order['end_grid'][:,None].astype(np.float)
        start_time=(order['start_time']//self.dispatch_interval)[:,None].astype(np.float)
        duration=(order['duration']//self.dispatch_interval)[:,None].astype(np.float)+1
        price=order['price'][:,None]
        self.real_orders=np.concatenate([start_grid,end_grid,start_time,duration,price],1)
        self.idle_driver_location_mat[0]= np.round(driver_locate_p*10000).astype(np.int)
        return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args= parser.parse_args()
    args.speed=3    #min/km
    args.driver_num=5000
    args.wait_time=10
    args.grid_num=500
    args.TIME_LEN=144
    dataset=kdd18(args)
    dataset.build_dataset()

    #dataset=map_regular(args)
    #neighbor=dataset.load_grid_neighbor()
    #dataset.process_order()
    #dataset.process_grid()
    #dataset.process_grid_neighbor()
    #dataset.load_grid_neighbor()
    #dataset.load_order()