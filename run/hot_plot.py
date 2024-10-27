from shapely.geometry import Point, LineString, Polygon, mapping
import numpy as np
import folium
from folium.features import DivIcon
from folium.plugins import HeatMap


# 获取六边形的边界和中心点，无需修改
# grid_hexagon : 六边形网格的中心点 shape = (36,2)
# poly_hexa  六边形网格的边界 shape = (36,6,2)
def get_hexagon():
    city = ['DIDI','NYU'][0]
    if city=='NYU':
        r=6371
        per_lon=r/360*2*3.14159*np.cos(np.pi*(40.5/180))
        per_lat=r/360*2*3.14159
        per_lonlat=np.array([per_lon,per_lat])
        corner=[
        [-74.04,40.91553],
        [-74.04,40.58 ],
        [-73.72 ,40.91553],
        [-73.72 ,40.58]
        ]
    elif city=='DIDI':
        r=6371
        per_lon=r/360*2*3.14159*np.cos(np.pi/6)
        per_lat=r/360*2*3.14159
        per_lonlat=np.array([per_lon,per_lat])
        corner=[
        [103.9521,30.7900],
        [103.9521,30.5400],
        [104.1934,30.7900],
        [104.1934,30.5400]
        ]

    def cal_dis(array1,array2):
        return np.sqrt(np.sum((np.abs(array1-array2)*per_lonlat)**2))

    # 绘制六边形的中心点
    corner=np.array(corner)
    radius=cal_dis(corner[0],corner[1])/13
    lat_num=int(cal_dis(corner[0],corner[1])/radius)
    lon_num=int(round(cal_dis(corner[0],corner[2])/(radius*np.sqrt(3)/2)))
    lat_per_radius=(corner[0,1]-corner[1,1])/lat_num
    lon_per_radius=(corner[2,0]-corner[0,0])/lon_num
    lat_num=6
    lon_num=6
    #print(radius, lat_num, lon_num, lat_per_radius, lon_per_radius)
    #assert lat_num==10 and lon_num==13 , print(lat_num,lon_num)
    grid_hexagon=np.zeros((lat_num,lon_num,2))
    center = corner[0]*3/4+corner[-1]/4
    for i in range(lat_num):
        for j in range(lon_num):
            grid_hexagon[i,j][0]=lon_per_radius*j+center[0]
            if j%2==0:
                grid_hexagon[i,j][1]=-lat_per_radius*(i+0.5)+center[1]
            else:
                grid_hexagon[i,j][1]=-lat_per_radius*i+center[1]

    grid_hexagon = grid_hexagon.reshape(-1,2)

    lon_per_radius,lat_per_radius = lon_per_radius*0.9, lat_per_radius*0.9

    def get_hexagon_poly(center):
        hexa = []   # [lon,lat]
        hexa.append([center[0]-lon_per_radius/3, center[1]+lat_per_radius/2])
        hexa.append([center[0]+lon_per_radius/3, center[1]+lat_per_radius/2])
        hexa.append([center[0]+lon_per_radius/3*2, center[1]])
        hexa.append([center[0]+lon_per_radius/3, center[1]-lat_per_radius/2])
        hexa.append([center[0]-lon_per_radius/3, center[1]-lat_per_radius/2])
        hexa.append([center[0]-lon_per_radius/3*2, center[1]])
        return hexa

    poly_hexa = np.array([get_hexagon_poly(g) for g in grid_hexagon])

    def rorate(data, theta):
        x,y = data[...,0], data[...,1]
        newx = np.cos(theta)*x-np.sin(theta)*y
        newy = np.cos(theta)*y+np.sin(theta)*x
        return np.stack([newx,newy],axis=-1)

    # 旋转网格
    poly_hexa = rorate(poly_hexa-np.mean(grid_hexagon,axis=0,keepdims=True)[:,None,:],np.pi/4)+np.mean(grid_hexagon,axis=0,keepdims=True)[:,None,:]
    grid_hexagon = rorate(grid_hexagon-np.mean(grid_hexagon,axis=0,keepdims=True),np.pi/4)+np.mean(grid_hexagon,axis=0,keepdims=True)
    #poly_hexa = [Polygon(get_hexagon_poly(g)) for g in grid_hexagon]

    grid_hexagon = grid_hexagon[...,[1,0]]
    poly_hexa = poly_hexa[...,[1,0]]
    return grid_hexagon,poly_hexa


#  获取订单分布, 无需修改
#  order_data : shape = (36,)
def get_order():
    np.random.seed(0)
    num_valid_grid = 36
    mapped_matrix_int = np.arange(num_valid_grid)
    mapped_matrix_int = np.reshape(mapped_matrix_int, (6, 6))
    central_node_ids = []
    M, N = mapped_matrix_int.shape
    order_num_dist = []
    idle_driver_location_mat = np.zeros((144, num_valid_grid))

    order_grid_param = np.random.randint(2, 10, num_valid_grid)

    for ii in np.arange(144):
        time_dict = {}
        random_disturb = np.random.randint(-2, 3, num_valid_grid)
        time_dict_param = order_grid_param + random_disturb
        time_dict_param[time_dict_param < 0] = 0
        # time_dict_param[time_dict_param<=2]=2
        for jj in np.arange(M * N):  # num of grids
            time_dict[jj] = [time_dict_param[jj].item()]  # mean is 2
        order_num_dist.append(time_dict)
        idle_driver_location_mat[ii, :] = [10] * num_valid_grid

    order_num = np.zeros(36, dtype=np.int32)
    for t in range(144):
        for i in range(36):
            order_num[i] += order_num_dist[t][i]

    return order_num


# 绘制六边形网格和热力图
grid_hexagon,poly_hexa = get_hexagon()  # 获取六边形网格
order_data = get_order()                # 获取订单分布
class_grid = np.load('best.npy', allow_pickle=True).tolist()  # 加载网格分类数据
order_data[class_grid[0]]=1315
order_data[class_grid[1]]=800
order_data[class_grid[2]]=500
order_data[class_grid[3]]=200
order_data[class_grid[0]]=order_data[class_grid[0]]+(np.random.random((len(class_grid[0]))) - 0.5)*100
order_data[class_grid[1]]=order_data[class_grid[1]]+(np.random.random((len(class_grid[1]))) - 0.5)*100
order_data[class_grid[2]]=order_data[class_grid[2]]+(np.random.random((len(class_grid[2]))) - 0.5)*100
order_data[class_grid[3]]=order_data[class_grid[3]]+(np.random.random((len(class_grid[3]))) - 0.5)*100

grid_order_data = np.concatenate([grid_hexagon,order_data[:,None]],axis=-1)   # 热力图的数据格式，[纬度,经度,订单数量]
# 将订单扩散到网格内部
'''
random_scatter = np.random.rand(36,20,6)
random_scatter = random_scatter/np.sum(random_scatter,axis=-1, keepdims=True)
grid_scatter = np.matmul(random_scatter, poly_hexa)
'''
'''
grid_scatter = np.concatenate([(poly_hexa+grid_hexagon[:,None,:])/2,grid_hexagon[:,None,:]],axis=1)
grid_order_data = np.concatenate([grid_scatter,np.tile(order_data[:,None,None],(1,7,1))],axis=-1)   # 热力图的数据格式，[纬度,经度,订单数量]
grid_order_data = grid_order_data.reshape((-1,3))
'''

draw_map=folium.Map(
    location=grid_hexagon.mean(0),
    zoom_start=20,
    control_scale=True,
    tiles='https://webrd02.is.autonavi.com/appmaptile?lang=en&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
    attr='高德-纯英文对照',)
folium.LatLngPopup().add_to(draw_map)
# 绘制六边形
for i in range(36):
    folium.Polygon(
        locations = poly_hexa[i] ,
        color='green',
        weight=1,
        fill_color='white',
        fillOpacity=0.1,
        popup=folium.Popup('grid:{}, OrderNum:{}'.format(i,order_data[i]))
        ).add_to(draw_map)
# 绘制热力图
HeatMap(grid_order_data).add_to(draw_map)
draw_map.save('热力图.html')
draw_map


# 绘制网格染色图
grid_hexagon,poly_hexa = get_hexagon()  # 获取六边形网格

class_grid=np.load("begin.npy",allow_pickle=True).tolist()
class_grid = np.load('best.npy', allow_pickle=True).tolist()  # 加载网格分类数据
SePS_grid={0:[5,11,17,4,10,16,2,8,1],1:[3,9,15,7,13,14,21,20],2:[0,6,12,18,24,30,19,31,32,26,],3:[25,23,33,34,35,28,29,22,27]}
class_grid_medium={0:[5,11,17,4,10,16,3,9,15],1:[2,8,1,7,13,14,21,20],2:[0,6,12,18,24,30,19,25,23],3:[31,32,26,33,34,35,28,29,22,27]}
class_grid=SePS_grid

# import numpy as np
#
# begin=np.load("begin.npy",allow_pickle=True).item()
# medium=np.load("midium.npy",allow_pickle=True).item()
# end=np.load("best.npy",allow_pickle=True).item()


# class_color = ['purple','red','blue','green']   # 定义类别颜色
#
# draw_map=folium.Map(
#     location=grid_hexagon.mean(0),
#     zoom_start=20,
#     control_scale=True,
#     tiles='https://webrd02.is.autonavi.com/appmaptile?lang=en&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
#     attr='高德-纯英文对照',)
# folium.LatLngPopup().add_to(draw_map)
# for i in range(4):
#     for id in class_grid[i]:
#         folium.Polygon(locations = poly_hexa[id] , color='red', weight=1,
#                 fill_color=class_color[i], fillOpacity=0.2,popup=folium.Popup('grid:'+str(i))).add_to(draw_map)
# draw_map.save('网格染色图.html')
