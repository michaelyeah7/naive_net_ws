import os
import logging
import sys
import socket
import numpy as np
import rospy
import torch
import torch.nn as nn
from mpi4py import MPI

from torch.optim import Adam
from collections import deque

from model2.net import MLPPolicy, CNNPolicy
from stage_world1_adv1 import StageWorld
#from turtlebot_world import TurtlebotWorld
from model2.ppo import ppo_update_stage1, generate_train_data
from model2.ppo import generate_action, generate_action_no_sampling
from model2.ppo import transform_buffer
import copy


MAX_EPISODES = 5000
LASER_BEAM = 360
LASER_HIST = 3
HORIZON = 128
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 1024
EPOCH = 2
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
NUM_ENV = 1
OBS_SIZE = 360
ACT_SIZE = 2
LEARNING_RATE = 5e-5



def run(env, policy, policy_path, action_bound, optimizer):

    # rate = rospy.Rate(5)
    buff = []
    global_update = 10
    global_step = 0


    # if env.index == 0:
    #     env.reset_world()

    print("whether run")
    for id in range(MAX_EPISODES):
        env.reset_pose()
        env.generate_goal_point()
        terminal = False
        ep_reward = 0
        step = 1
        obs = env.get_laser_observation()
        obs_stack = deque([obs, obs, obs])
        goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        state = [obs_stack, goal, speed]
        while not terminal and not rospy.is_shutdown():
            #state_list = comm.gather(state, root=0)
            state_list = [state]
            #print("state_list.len",len(state_list))

            # generate actions at rank==0
            v, a, logprob, scaled_action=generate_action(env=env, state_list=state_list,
                                                         policy=policy, action_bound=action_bound)
            # _, scaled_action=generate_action_no_sampling(env=env, state_list=state_list,
            #                                              policy=policy, action_bound=action_bound)
            # execute actions
            #real_action = comm.scatter(scaled_action, root=0)
            real_action = copy.deepcopy(scaled_action[0])
            env.control_vel(real_action)
            # rate.sleep()
            rospy.sleep(0.02)

            # get informtion
            r, terminal, result = env.get_reward_and_terminate(step)
            ep_reward += r
            global_step += 1


            # get next state
            s_next = env.get_laser_observation()
            left = obs_stack.popleft()
            obs_stack.append(s_next)
            goal_next = np.asarray(env.get_local_goal())
            speed_next = np.asarray(env.get_self_speed())
            state_next = [obs_stack, goal_next, speed_next]

            if global_step % HORIZON == 0:
                #state_next_list = comm.gather(state_next, root=0)
                state_next_list = [state_next]
                
                last_v, _, _, _ = generate_action(env=env, state_list=state_next_list, policy=policy,
                                                               action_bound=action_bound)
            # add transitons in buff and update policy
            #r_list = comm.gather(r, root=0)
            r_list = [r]
            #terminal_list = comm.gather(terminal, root=0)
            terminal_list = [terminal]

            if env.index == 0:
                buff.append((state_list, a, r_list, terminal_list, logprob, v))
                if len(buff) > HORIZON - 1:
                    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                        transform_buffer(buff=buff)
                    t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                            last_value=last_v, dones=d_batch, lam=LAMDA)
                    memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                    ppo_update_stage1(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                                            epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                                            num_env=NUM_ENV, frames=LASER_HIST,
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE)
                    buff = []
                    global_update += 1

            step += 1
            state = state_next


        if env.index == 1:
            if global_update != 0 and global_update % 20 == 0:
                torch.save(policy.state_dict(), policy_path + '/Stage1_Ad1_{}'.format(global_update))
                logger.info('########################## model saved when update {} times#########'
                            '################'.format(global_update))
        #distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)
        distance = 0
        logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, Reward %-5.1f, Distance %05.1f, %s' % \
                    (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, ep_reward, distance, result))
        logger_cal.info(ep_reward)





if __name__ == '__main__':
    # config log
    hostname = socket.gethostname()
    if not os.path.exists('./log_adv1_can_use/' + hostname):
        os.makedirs('./log_adv1_can_use/' + hostname)
    output_file = './log_adv1_can_use/' + hostname + '/output.log'
    cal_file = './log_adv1_can_use/' + hostname + '/cal.log'

    # config log
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger_cal = logging.getLogger('loggercal')
    logger_cal.setLevel(logging.INFO)
    cal_f_handler = logging.FileHandler(cal_file, mode='a')
    file_handler.setLevel(logging.INFO)
    logger_cal.addHandler(cal_f_handler)

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()

    

    #env = TurtlebotWorld(360, index=rank, num_env=NUM_ENV)
    env = StageWorld(OBS_SIZE, index=0, num_env=NUM_ENV)

    reward = None
    action_bound = [[0, -1], [0.5, 1]]

    # torch.manual_seed(1)
    # np.random.seed(1)

    policy_path = 'policy/adv1_can_use'
    # policy = MLPPolicy(obs_size, act_size)
    policy = CNNPolicy(frames=LASER_HIST, action_space=2)
    policy.cuda()
    opt = Adam(policy.parameters(), lr=LEARNING_RATE)
    mse = nn.MSELoss()

    if not os.path.exists(policy_path):
        os.makedirs(policy_path)

    file = policy_path + '/Stage1_800_real'
    if os.path.exists(file):
        logger.info('####################################')
        logger.info('############Loading Model###########')
        logger.info('####################################')
        state_dict = torch.load(file)
        policy.load_state_dict(state_dict)
    else:
        logger.info('#####################################')
        logger.info('############Start Training###########')
        logger.info('#####################################')


    try:
        run(env=env, policy=policy, policy_path=policy_path, action_bound=action_bound, optimizer=opt)
    except KeyboardInterrupt:
        pass