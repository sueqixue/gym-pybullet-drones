"""The simulation is run by a `FLabCtrlAviary` environment.

Example
-------
In a terminal, run as:

    $ python traj_opt.py

Notes
-----
Trajetory generation.

Implemented by Qi Xue (qixue@seas.upenn.edu).

"""
import numpy as np
from copy import deepcopy
from rrt_drones import * 

""" Collision avoidance algorithms
        none - dummy trajectory
        rrt - rrt algorithm
        pp - potential field
"""
DUMMY = 'none'
RRT = 'rrt'
PP = 'pp'

# Debug boolens
PRINTING = False
RRT_PRINTING = False
TP_PRINTING = True

"""
Taking off the drone to the hover xyz and move to the dest position.
Note that TARGET_POS is an array contains the position at each sampling
point, which means it contains the infomation of the desired trajectory/
path of the current task. So when implementing any collision avoidance
algorithms/controller, use the input src and dest position to find the
optimal trajectory/path.
"""
def traj_opt(INIT_XYZ, 
            HOVER_XYZ, 
            DEST_XYZ,
            drone_origin_pos,
            drone_origin_ori,
            env,
            GROUND_EFFECT=True,
            TAKEOFF_PERIOD=8,
            TASK_PERIOD=12,
            HOVER_PERIOD=4,
            control_freq_hz=240,
            collision_avoidance=RRT,
            take_off_flag=False
            ):
   
    TAKE_OFF_PAR = 0.15*3
    TAKE_OFF_H = HOVER_XYZ[0, 2]
    HOVER_FLAG = False

    # Take off wp
    if take_off_flag:
        NUM_WP_TAKEOFF = control_freq_hz*TAKEOFF_PERIOD
    else:
        NUM_WP_TAKEOFF = 0
    # Task wp
    NUM_WP_TASK = control_freq_hz*TASK_PERIOD
    # Hover wp
    if take_off_flag: 
        NUM_WP_HOVER = 0
    else:
        NUM_WP_HOVER = control_freq_hz*HOVER_PERIOD
    # Total wp
    NUM_WP = NUM_WP_TAKEOFF + NUM_WP_TASK + NUM_WP_HOVER

    TARGET_POS = np.zeros((NUM_WP,3))

    # Take off
    if take_off_flag:
        if GROUND_EFFECT:
            print(f"\n---------- TAKE OFF WITH GROUND EFFECT ----------\n")
        else:
            print(f"\n---------- TAKE OFF WITHOUT GROUND EFFECT ----------\n")

        for i in range(NUM_WP_TAKEOFF):
            if GROUND_EFFECT:
                if not HOVER_FLAG:
                    TARGET_POS[i, :] = INIT_XYZ[0, 0], INIT_XYZ[0, 1], INIT_XYZ[0, 2] + TAKE_OFF_PAR * (np.sin((i/NUM_WP_TAKEOFF)*(2*np.pi)) + 1)
                else:
                    TARGET_POS[i, :] = INIT_XYZ[0, 0], INIT_XYZ[0, 1], TARGET_POS[i-1, 2]

                if TARGET_POS[i, 2] < TARGET_POS[i-1, 2]:
                    HOVER_FLAG = True
                    NUM_WP_TAKEOFF = i+1
            else:
                TARGET_POS[i, :] = INIT_XYZ[0, 0], INIT_XYZ[0, 1], INIT_XYZ[0, 2] + i * (TAKE_OFF_H/NUM_WP_TAKEOFF)
            
            if TP_PRINTING:
                if i % 100 == 0:
                    print(f"TARGET_POS[{i}, :] = {TARGET_POS[i, :]}")
        
        task_start = TARGET_POS[NUM_WP_TAKEOFF-1, :].reshape(1,3)
    else:
        task_start = INIT_XYZ

    # Finding the path to the destination
    task_goal = DEST_XYZ
    task_path = []
    smooth_task_path = []
    if collision_avoidance == 'rrt':
        print(f"\n---------- RRT ----------\n")
        task_path = rrt(deepcopy(env), deepcopy(task_start), deepcopy(task_goal), NUM_WP_TASK)
        task_path_len = len(task_path)
        if PRINTING:
            print(f"task_path = {task_path}")
            print(f"RRT path length = {task_path_len}")
            print(f"RRT path shape = {task_path.shape}")
        
        # Smooth the path
        smooth_task_path = np.zeros((NUM_WP_TASK,3))
        if RRT_PRINTING:
            print(f"smooth_task_path shape is {smooth_task_path.shape}")
        if task_path_len < NUM_WP_TASK:
            part_len = NUM_WP_TASK // (task_path_len - 1)
            if RRT_PRINTING:
                print(f"part_len = {part_len}\n")
                
            for i in range(task_path_len-1):
                if RRT_PRINTING:
                    print(f"section {i}:\n")
                
                sec_index = i*part_len
                smooth_task_path[sec_index] = task_path[i]
                if RRT_PRINTING:
                    print(f"smooth_task_path[{sec_index}] = {smooth_task_path[sec_index]}")
                
                for j in range(1, part_len):
                    step_diff_x = (task_path[i+1, 0] - task_path[i, 0]) / part_len
                    step_diff_y = (task_path[i+1, 1] - task_path[i, 1]) / part_len
                    step_diff_z = (task_path[i+1, 2] - task_path[i, 2]) / part_len
                    smooth_task_path[sec_index+j] = smooth_task_path[sec_index+j-1] + [step_diff_x, step_diff_y, step_diff_z]
                    if RRT_PRINTING:
                        if sec_index+j % 100 == 0:
                            print(f"smooth_task_path[{sec_index+j}] = {smooth_task_path[sec_index+j]}")

    if GROUND_EFFECT:
        print(f"\n---------- FLY TO DESTINATION WITH GROUND EFFECT ----------\n")
    else:
        print(f"\n---------- FLY TO DESTINATION WITHOUT GROUND EFFECT ----------\n")

    # Reset the pos and ori of the drone after collision detection
    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], drone_origin_pos, drone_origin_ori, env.CLIENT)

    if collision_avoidance == 'none':
        # [DEBUG]: Generating the path to the destination without collision avoidance
        print(f"\n---------- DEBUG TRAJ ----------\n")
        for i in range(NUM_WP_TASK):
            TARGET_POS[i+NUM_WP_TAKEOFF, :] = TARGET_POS[NUM_WP_TAKEOFF-1, 0] + i * (TAKE_OFF_H/NUM_WP_TASK), TARGET_POS[NUM_WP_TAKEOFF-1, 1], TARGET_POS[NUM_WP_TAKEOFF-1, 2]
            if TP_PRINTING:
                if i % 100 == 0:
                    print(f"TARGET_POS[{i+NUM_WP_TAKEOFF}, :] = {TARGET_POS[i+NUM_WP_TAKEOFF, :]}")
    elif collision_avoidance == 'rrt':
        for i in range(NUM_WP_TASK):
            if i < len(smooth_task_path):
                TARGET_POS[i+NUM_WP_TAKEOFF, :] = smooth_task_path[i]
            elif len(smooth_task_path) > 0:
                TARGET_POS[i+NUM_WP_TAKEOFF, :] = smooth_task_path[-1]
            else:
                TARGET_POS[i+NUM_WP_TAKEOFF, :] = TARGET_POS[NUM_WP_TAKEOFF-1, :]

            if TP_PRINTING:
                if i % 100 == 0:
                    print(f"TARGET_POS[{i+NUM_WP_TAKEOFF}, :] = TASK_POS[{i}, :] = {TARGET_POS[i+NUM_WP_TAKEOFF, :]}")
    
    if TP_PRINTING:
        print(f"TARGET_POS[{NUM_WP_TASK+NUM_WP_TAKEOFF-1}, :] = {TARGET_POS[NUM_WP_TASK+NUM_WP_TAKEOFF-1, :]}")

    if not take_off_flag:
        if GROUND_EFFECT:
            print(f"\n---------- HOVER WITH GROUND EFFECT ----------\n")
        else:
            print(f"\n---------- HOVER WITHOUT GROUND EFFECT ----------\n")

    for i in range(NUM_WP_HOVER):
        TARGET_POS[i+NUM_WP_TAKEOFF+NUM_WP_TASK, :] = TARGET_POS[i+NUM_WP_TAKEOFF+NUM_WP_TASK-1, :]

        if TP_PRINTING:
                if i % 100 == 0:
                    print(f"TARGET_POS[{i+NUM_WP_TAKEOFF+NUM_WP_TASK}, :] = {TARGET_POS[i+NUM_WP_TAKEOFF+NUM_WP_TASK, :]}")

    actual_num = NUM_WP_TAKEOFF + NUM_WP_TASK + NUM_WP_HOVER
    if PRINTING:
        print(f"actual_num = {actual_num}")
        print(f"NUM_WP = {NUM_WP}")

    if actual_num < NUM_WP:
        NUM_WP = actual_num
        # for i in range(actual_num, NUM_WP):
        #     TARGET_POS[i, :] = TARGET_POS[i-1, :]

        #     if TP_PRINTING:
        #         if i % 100 == 0:
        #             print(f"TARGET_POS[{i}, :] = {TARGET_POS[i, :]}")
    
    return TARGET_POS[:actual_num, :], NUM_WP