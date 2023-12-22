import csv
import numpy as np
import pandas as pd

row_num = 0


def convert(input_x, input_y, input_z, output):
        global row_num
        x = pd.read_csv(input_x, sep=',', parse_dates=False).values
        y = pd.read_csv(input_y, sep=',', parse_dates=False).values
        z = pd.read_csv(input_z, sep=',', parse_dates=False).values
        row_num = x.shape[0]
        comb = np.zeros((x.shape[0], 3))
        comb[:, 0] = x[:, 1]
        comb[:, 1] = y[:, 1]
        comb[:, 2] = z[:, 1]
        pd.DataFrame(comb).to_csv(output)


def obst_state_convert(obst_num):
        obst = np.zeros((row_num, obst_num, 4, 3))

        for i in range(obst_num):
                pos_csv = 'obst_pos_' + str(i) + '.csv'
                orit_csv = 'obst_orit_' + str(i) + '.csv'
                vel_csv = 'obst_vel_' + str(i) + '.csv'
                ang_vel_csv = 'obst_ang_vel_' + str(i) + '.csv'

                pos = pd.read_csv(pos_csv, sep=',', parse_dates=False).values[:, 1:4]
                orit = pd.read_csv(orit_csv, sep=',', parse_dates=False).values[:, 1:4]
                vel = pd.read_csv(vel_csv, sep=',', parse_dates=False).values[:, 1:4]
                ang_vel = pd.read_csv(ang_vel_csv, sep=',', parse_dates=False).values[:, 1:4]

                obst_i = np.zeros((pos.shape[0], 4, 3))
                obst_i[:, 0] = pos[:,]
                obst_i[:, 1] = orit[:,]
                obst_i[:, 2] = vel[:,]
                obst_i[:, 3] = ang_vel[:,]

                obst[:, i] = obst_i


def get_dy_obst_states(obst_num):
        convert('data/x0.csv', 'data/y0.csv', 'data/z0.csv', 'obst_pos_0.csv')
        convert('data/x1.csv', 'data/y1.csv', 'data/z1.csv', 'obst_pos_1.csv')
        convert('data/x2.csv', 'data/y2.csv', 'data/z2.csv', 'obst_pos_2.csv')

        convert('data/r0.csv', 'data/p0.csv', 'data/y0.csv', 'obst_pos_0.csv')
        convert('data/r1.csv', 'data/p1.csv', 'data/y1.csv', 'obst_orit_1.csv')
        convert('data/r2.csv', 'data/p2.csv', 'data/y2.csv', 'obst_orit_2.csv')

        convert('data/vx0.csv', 'data/vy0.csv', 'data/vz0.csv', 'obst_pos_0.csv')
        convert('data/vx1.csv', 'data/vy1.csv', 'data/vz1.csv', 'obst_vel_1.csv')
        convert('data/vx2.csv', 'data/vy2.csv', 'data/vz2.csv', 'obst_vel_2.csv')

        convert('data/wx0.csv', 'data/wy0.csv', 'data/wz0.csv', 'obst_ang_vel_0.csv')
        convert('data/wx1.csv', 'data/wy1.csv', 'data/wz1.csv', 'obst_ang_vel_1.csv')
        convert('data/wx2.csv', 'data/wy2.csv', 'data/wz2.csv', 'obst_ang_vel_2.csv')

        print(f"row_num = {row_num}")

        return obst_state_convert(obst_num)

obst_num = 3
dy_obst_state = get_dy_obst_states(obst_num)
np.save('dy_obst_state', dy_obst_state)