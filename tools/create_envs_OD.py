import numpy as np
from simulator.envs import *
from parse_data import *
import os.path as osp

def create_env(args):

    dataset=kdd18(args)
    dataset.build_dataset(args)

    n_side = 6
    l_max = 2

    env = CityReal(dataset.mapped_matrix_int, dataset.order_num_dist, dataset.idle_driver_dist_time, dataset.idle_driver_location_mat,
                   dataset.order_time, dataset.order_price, l_max , dataset.M, dataset.N, n_side, dataset.TIME_LEN, dataset.order_ratio, dataset.order_real,
                   dataset.onoff_driver_location_mat, fleet_help=args.fleet_help,time_interval=int(1440//dataset.TIME_LEN),grid_neighbor=dataset.grid_neighbor)

    return env, dataset.M, dataset.N, [], len(dataset.mapped_matrix_int)