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

from model.net import MLPPolicy, CNNPolicy, AdversaryPush_CNNPolicy
from stage_ad_notPass_naive_net import StageWorld
from model.ppo import ppo_update_stage1, generate_train_data,ppo_update_adversayPush
from model.ppo import generate_action,generate_action_adPush
from model.ppo import transform_buffer,transform_buffer_adPush



MAX_EPISODES = 5000
EP_LEN = 400
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
TRAIN = True


def run(env, policy, policy_path, action_bound, optimizer):

    # rate = rospy.Rate(5)
    buff = []
    global_update = 0
    global_step = 0


    if env.index == 0:
        env.reset_world()
    env.store_resetPose()


    for id in range(MAX_EPISODES):
        env.reset_pose()

        # env.generate_goal_point()
        terminal = False
        next_ep = False
        ep_reward = 0
        step = 1

        # obs = env.get_laser_observation()
        # obs_stack = deque([obs, obs, obs])
        ad_speed = np.asarray(env.get_self_speed())
        ag_pos_and_goal = np.asarray(env.get_local_ag_pos_and_goal())

        state = [ad_speed, ag_pos_and_goal]

        while not terminal and not rospy.is_shutdown():
            #state_list = comm.gather(state, root=0)
            state_list = [state]


            # generate actions at rank==0
            v, a, logprob, scaled_action=generate_action_adPush(env=env, state_list=state_list,
                                                         policy=policy, action_bound=action_bound)

            # execute actions
            #real_action = comm.scatter(scaled_action, root=0)
            real_action = scaled_action[0]
            env.control_vel(real_action)

            # rate.sleep()
            rospy.sleep(0.001)

            # get informtion
            r, terminal, result = env.get_reward_and_terminate(step)
            ep_reward += r
            global_step += 1

            # if (terminal):
            #     env.reset_pose()
            # if(step > EP_LEN):
            #     next_ep = True


            # get next state
            # s_next = env.get_laser_observation()
            # left = obs_stack.popleft()
            # obs_stack.append(s_next)
            # goal_next = np.asarray(env.get_local_goal())
            # speed_next = np.asarray(env.get_self_speed())
            # state_next = [obs_stack, goal_next, speed_next]

            ad_speed = np.asarray(env.get_self_speed())
            ag_pos_and_goal = np.asarray(env.get_local_ag_pos_and_goal())
            state_next = [ad_speed, ag_pos_and_goal]

            if global_step % HORIZON == 0:
                #state_next_list = comm.gather(state_next, root=0)
                state_next_list = [state_next]
                last_v, _, _, _ = generate_action_adPush(env=env, state_list=state_next_list, policy=policy,
                                                               action_bound=action_bound)
            # add transitons in buff and update policy
            #r_list = comm.gather(r, root=0)
            r_list = [r]
            #terminal_list = comm.gather(terminal, root=0)
            terminal_list = [terminal]

            if env.mpi_rank == 0:
                buff.append((state_list, a, r_list, terminal_list, logprob, v))
                if len(buff) > HORIZON - 1:
                    goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                        transform_buffer_adPush(buff=buff)
                    t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                              last_value=last_v, dones=d_batch, lam=LAMDA)
                    memory = (goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                    ppo_update_adversayPush(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                                            epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                                            num_env=NUM_ENV, frames=LASER_HIST,
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE)

                    buff = []
                    global_update += 1

            step += 1
            state = state_next


        if env.mpi_rank == 0:
            if TRAIN:
                if global_update != 0 and global_update % 20 == 0:
                    torch.save(policy.state_dict(), policy_path + '/Stage1_{}'.format(global_update))
                    logger.info('########################## model saved when update {} times#########'
                                '################'.format(global_update))
        # distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)
        #distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)
        distance = 0

        if TRAIN:
            logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, Reward %-5.1f, Distance %05.1f, %s' % \
                        (env.mpi_rank, 0, 3, id + 1, step, ep_reward, distance, result))
            logger_cal.info(ep_reward)





if __name__ == '__main__':
    ROS_PORT0 = 11325
    NUM_BOT = 4 #num of robot per stage
    NUM_ENV = 1 #num of env(robots)
    ID = 1014 #policy saved directory
    ROBO_START = 1 #ad robtos start index
    GOAL_START = 0 # goal robot start index
    TRAIN = True
    POLICY_NAME = "/Stage1_12180"

    # config log
    # hostname = socket.gethostname()
    # hostname = "autoRL_%d"%ID
    # dirname = '/clever/saved_model_ppo/' + hostname
    # logdir = dirname + "/log"
    # policydir = dirname
    # if not os.path.exists(dirname):
    #     os.makedirs(dirname)
    # if not os.path.exists(logdir):
    #     os.makedirs(logdir)
    # output_file =logdir + '/output.log'
    # cal_file = logdir + '/cal.log'
    hostname = socket.gethostname()
    if not os.path.exists('./log_adv2_naive_network/' + hostname):
        os.makedirs('./log_adv2_naive_network/' + hostname)
    output_file = './log_adv2_naive_network/' + hostname + '/output.log'
    cal_file = './log_adv2_naive_network/' + hostname + '/cal.log'

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
    # rosPort = ROS_PORT0 
    # robotIndex = ROBO_START + (rank%NUM_BOT)
    # envIndex =  int(rank/NUM_BOT)
    # rosPort = rosPort + int(rank/NUM_BOT)
    # logger.info('rosport: %d robotIndex: %d rank:%d' %(rosPort,robotIndex,rank))


    env = StageWorld(beam_num=360, index=1, num_env=NUM_ENV,ros_port = 0,mpi_rank = 0,env_index = 0,goal_robotIndex=GOAL_START)
    reward = None
    action_bound = [[0, -1], [1, 1]]#[0.7, 1]]

    # torch.manual_seed(1)
    # np.random.seed(1)
    rank = 0
    if rank == 0:
        #policy_path = policydir
        policy_path = 'policy/ad_notPass'
        # policy = MLPPolicy(obs_size, act_size)
        #policy = CNNPolicy(frames=LASER_HIST, action_space=2)
        policy = AdversaryPush_CNNPolicy(frames=LASER_HIST, action_space=2)
        policy.cuda()
        opt = Adam(policy.parameters(), lr=LEARNING_RATE)
        mse = nn.MSELoss()

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        file = policy_path + POLICY_NAME
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
    else:
        policy = None
        policy_path = None
        opt = None

    try:
        run(env=env, policy=policy, policy_path=policy_path, action_bound=action_bound, optimizer=opt)
    except KeyboardInterrupt:
        pass