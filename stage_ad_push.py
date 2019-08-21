import time
import rospy
import copy
import tf
import numpy as np
import os
import random


from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from std_msgs.msg import Int8


class StageWorld():
    def __init__(self, beam_num, index, num_env,ros_port,mpi_rank,env_index,goal_robotIndex):
        os.environ["ROS_MASTER_URI"]="http://localhost:%d"%ros_port
        self.mpi_rank =mpi_rank
        self.index = index
        self.num_env = num_env
        self.env_index = env_index
        node_name = 'StageEnv_' + str(index)
        print("rank: %d node name:%s"%(mpi_rank,node_name))
        rospy.init_node(node_name, anonymous=None)

        self.beam_mum = beam_num
        self.laser_cb_num = 0
        self.scan = None

        # used in reset_world
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.

        # used in generate goal point
        self.map_size = np.array([8., 8.], dtype=np.float32)  # 20x20m
        self.goal_size = 0.5

        self.robot_value = 10.
        self.goal_value = 0.
        # self.reset_pose = None



        # for get reward and terminate
        self.stop_counter = 0

        # -----------Publisher and Subscriber-------------
        cmd_vel_topic = 'robot_' + str(index) + '/cmd_vel'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        cmd_pose_topic = 'robot_' + str(index) + '/cmd_pose'
        self.cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=2)

        object_state_topic = 'robot_' + str(index) + '/base_pose_ground_truth'
        self.object_state_sub = rospy.Subscriber(object_state_topic, Odometry, self.ground_truth_callback)

        laser_topic = 'robot_' + str(index) + '/base_scan'

        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_scan_callback)

        odom_topic = 'robot_' + str(index) + '/odom'
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)

        crash_topic = 'robot_' + str(index) + '/is_crashed'
        self.check_crash = rospy.Subscriber(crash_topic, Int8, self.crash_callback)


        self.sim_clock = rospy.Subscriber('clock', Clock, self.sim_clock_callback)

        self.goal_roboIndex = goal_robotIndex
        goalRobo_goal_topic = 'robot_' + str(self.goal_roboIndex) + '/goal_pose'
        self.goalRobo_goal_sub = rospy.Subscriber(goalRobo_goal_topic, Pose, self.goalRobo_goal_callback)
        self.goal_point = [0,0]

        goalRobo_odom_topic = 'robot_' + str(self.goal_roboIndex) + '/odom'
        self.goalRobo_odom_sub = rospy.Subscriber(goalRobo_odom_topic, Odometry, self.goalRobo_odometry_callback)

        # goalRobo_speed_topic = 'robot_' + str(self.goal_roboIndex) + '/cmd_vel'
        # self.goalRobo_speed_sub = rospy.Subscriber(goalRobo_speed_topic, Twist, self.goalRobo_speed_callback)
        # -----------Service-------------------
        self.reset_stage = rospy.ServiceProxy('reset_positions', Empty)

        # # get initial pose for resetting
        self.odom_topic = odom_topic
        self.first_pose = None
        while self.first_pose is None:
            try:
                self.first_pose = rospy.wait_for_message(odom_topic, Odometry, timeout=5).pose.pose
            except:
                pass
        #for compute distance
        self.init_pose = [self.first_pose.position.x, self.first_pose.position.y]

        # # Wait until the first callback
        self.speed = None
        self.state = None
        self.speed_GT = None
        self.state_GT = None
        self.goal_roboPos = [0,0,0]
        self.speed_goalRobo = [0,0]
        self.ag_distance = 0
        while self.scan is None or self.speed is None or self.state is None\
                or self.speed_GT is None or self.state_GT is None:
            pass

        rospy.sleep(1.)
        # # What function to call when you ctrl + c
        # rospy.on_shutdown(self.shutdown)


    def ground_truth_callback(self, GT_odometry):
        Quaternious = GT_odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w])
        self.state_GT = [GT_odometry.pose.pose.position.x, GT_odometry.pose.pose.position.y, Euler[2]]
        v_x = GT_odometry.twist.twist.linear.x
        v_y = GT_odometry.twist.twist.linear.y
        v = np.sqrt(v_x**2 + v_y**2)
        self.speed_GT = [v, GT_odometry.twist.twist.angular.z]

    def laser_scan_callback(self, scan):
        self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
                           scan.scan_time, scan.range_min, scan.range_max]
        self.scan = np.array(scan.ranges)
        self.laser_cb_num += 1


    def odometry_callback(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def goalRobo_goal_callback(self, goal):
        self.goal_point = [goal.position.x, goal.position.y]
        [x, y] = self.get_local_goal()
        self.pre_distance = np.sqrt(x ** 2 + y ** 2)
        self.pre_ag_distance = np.sqrt((self.goal_point[0] - self.goal_roboPos[0]) ** 2 + (self.goal_point[1] - self.goal_roboPos[1]) ** 2)
        self.distance = copy.deepcopy(self.pre_distance)
        self.ag_distance = copy.deepcopy(self.pre_ag_distance)
        print("get goal:",self.goal_point)
        self.first_pose = goal
        self.first_pose.orientation.z = 1.0


    def goalRobo_odometry_callback(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.goal_roboPos = [odometry.pose.pose.position.x, odometry.pose.pose.position.y,Euler]
        self.speed_goalRobo = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]


    def sim_clock_callback(self, clock):
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.

    def crash_callback(self, flag):
        self.is_crashed = flag.data

    def get_self_stateGT(self):
        return self.state_GT

    def get_self_speedGT(self):
        return self.speed_GT

    def get_laser_observation(self):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 3.5
        scan[np.isinf(scan)] = 3.5
        raw_beam_num = len(scan)
        sparse_beam_num = self.beam_mum
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []
        index = 0.
        for x in xrange(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan[int(index)])
            index += step
        sparse_scan_right = []
        index = raw_beam_num - 1.
        for x in xrange(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index -= step
        scan_sparse = np.concatenate((sparse_scan_right[::-1],sparse_scan_left), axis=0)
        return scan_sparse / 3.5 - 0.5


    def get_self_speed(self):
        return self.speed

    def get_self_state(self):
        return self.state

    def get_crash_state(self):
        return self.is_crashed

    def get_sim_time(self):
        return self.sim_time

    def get_local_goal(self):
        [x, y, theta] = self.get_self_stateGT()
        [goal_x, goal_y] = self.goal_point
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]

    def reset_world(self):
        self.reset_stage()
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.
        self.start_time = time.time()
        rospy.sleep(0.5)
        self.first_pose = None
        while self.first_pose is None:
            try:
                # print("waiting")
                self.first_pose = rospy.wait_for_message(self.odom_topic, Odometry, timeout=5).pose.pose
            except:
                pass

    def store_resetPose(self):
        self.first_pose = None
        while self.first_pose is None:
            try:
                # print("waiting")
                self.first_pose = rospy.wait_for_message(self.odom_topic, Odometry, timeout=5).pose.pose
            except:
                pass      


    def generate_goal_point(self):
        # goal point is assigned in the callback
        return


    def get_reward_and_terminate(self, t):
        terminate = False
        laser_scan = self.get_laser_observation()
        [x, y, theta] = self.get_self_stateGT()
        [v, w] = self.get_self_speedGT()
        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)
        self.pre_agDistance = copy.deepcopy(self.ag_distance)
        self.ag_distance = np.sqrt((self.goal_point[0] - self.goal_roboPos[0]) ** 2 + (self.goal_point[1] - self.goal_roboPos[1]) ** 2)
        # reward_g = (self.pre_distance - self.distance) * 2.5
        #reward setting refered by openAI's particles' envs
        reward_ad_g  = self.pre_distance - self.distance
        reward_ag_g  = self.pre_agDistance - self.ag_distance
        reward_g = (reward_ad_g - reward_ag_g) * 2.5
        reward_c = 0
        reward_w = 0
        result = 0
        is_crash = self.get_crash_state()

        # if is_crash == 1:
        #     terminate = True
        #     reward_c = -15
        #     result = 'Crashed'

        if is_crash == 1:
            if self.distance < self.goal_size:
                terminate = True
                result = 'goal Crashed'
            else:
                terminate = True
                reward_c = -15
                result = 'self Crashed'

        if self.distance < self.goal_size:
            reward_g = 15
            result = 'Reach Goal'

        if np.abs(w) >  1.05:
            reward_w = -0.1 * np.abs(w)

        if t > 200:
            result = 'Time out'

        reward = reward_g + reward_c + reward_w

        # print("my goal:",self.goal_point,"my reward_g",reward_g)

        return reward, terminate, result

    def reset_pose(self):
        # random_pose = self.generate_random_pose()
        ## reset robot pose to the initial position
        first_pose = self.first_pose
        rospy.sleep(0.01)
        self.cmd_pose.publish(first_pose)
        rospy.sleep(0.02)
        #------recalculate the distance when reset pose------#
        [x, y] = self.get_local_goal()
        self.pre_distance = np.sqrt(x ** 2 + y ** 2)
        self.distance = copy.deepcopy(self.pre_distance)
        # self.control_pose(random_pose)
        # [x_robot, y_robot, theta] = self.get_self_stateGT()

        # # start_time = time.time()
        # while np.abs(random_pose[0] - x_robot) > 0.2 or np.abs(random_pose[1] - y_robot) > 0.2:
        #     [x_robot, y_robot, theta] = self.get_self_stateGT()
        #     self.control_pose(random_pose)
        # rospy.sleep(0.01)


    def control_vel(self, action):
        move_cmd = Twist()
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]
        self.cmd_vel.publish(move_cmd)


    def control_pose(self, pose):
        pose_cmd = Pose()
        assert len(pose)==3
        pose_cmd.position.x = pose[0]
        pose_cmd.position.y = pose[1]
        pose_cmd.position.z = 0

        qtn = tf.transformations.quaternion_from_euler(0, 0, pose[2], 'rxyz')
        pose_cmd.orientation.x = qtn[0]
        pose_cmd.orientation.y = qtn[1]
        pose_cmd.orientation.z = qtn[2]
        pose_cmd.orientation.w = qtn[3]
        self.cmd_pose.publish(pose_cmd)

    def generate_random_pose(self):
        x = np.random.uniform(-9, 9)
        y = np.random.uniform(-9, 9)
        dis = np.sqrt(x ** 2 + y ** 2)
        while (dis > 9) and not rospy.is_shutdown():
            x = np.random.uniform(-9, 9)
            y = np.random.uniform(-9, 9)
            dis = np.sqrt(x ** 2 + y ** 2)
        theta = np.random.uniform(0, 2 * np.pi)
        return [x, y, theta]

    def generate_random_goal(self):
        self.init_pose = self.get_self_stateGT()
        x = np.random.uniform(-9, 9)
        y = np.random.uniform(-9, 9)
        dis_origin = np.sqrt(x ** 2 + y ** 2)
        dis_goal = np.sqrt((x - self.init_pose[0]) ** 2 + (y - self.init_pose[1]) ** 2)
        while (dis_origin > 9 or dis_goal > 10 or dis_goal < 8) and not rospy.is_shutdown():
            x = np.random.uniform(-9, 9)
            y = np.random.uniform(-9, 9)
            dis_origin = np.sqrt(x ** 2 + y ** 2)
            dis_goal = np.sqrt((x - self.init_pose[0]) ** 2 + (y - self.init_pose[1]) ** 2)

        return [x, y]

    def generate_stage_goal(self):
        env0_goals = [(2,0),(2,8),(-4,0),(-4,3),(-4,-2),(-5,-6),(7,-4),(7,-6)]
        env1_goals = []
        env2_goals = [(5.5,3),(8,2),(0,3),(0,6),(-3,0),(-7,0),(1,-4),(1,-6)]
        env3_goals = [(-2,2),(-2,6),(3,2),(3,6),(-2,-2),(-2,-6),(3,-3),(7,-7)]
        env4_goals = [(3,1),(7,1),(1,1),(5,1),(5,-1.5),(7,-1.5),(4,-4),(7.5,6.5)]
        env5_goals = [(0,1),(0,2),(4,-1),(8,-6.5),(0,-7.5),(4,-1),(8,-6.5),(0,-7.5)]
        goal_index = random.randrange(0, 8)     
        if(self.env_index == 0):
            x = env0_goals[goal_index][0]
            y = env0_goals[goal_index][1]
            return [x,y]
        elif (self.env_index ==1):
            return self.generate_random_goal()
        elif (self.env_index ==2 ):
            x = env2_goals[goal_index][0]
            y = env2_goals[goal_index][1]
            return [x,y]        
        elif (self.env_index ==3 ):
            x = env3_goals[goal_index][0]
            y = env3_goals[goal_index][1]
            return [x,y]
        elif (self.env_index ==4 ):
            x = env4_goals[goal_index][0]
            y = env4_goals[goal_index][1]
            return [x,y]
        elif (self.env_index ==5 ):
            x = env5_goals[goal_index][0]
            y = env5_goals[goal_index][1]
            return [x,y]

    def get_local_goalRobo(self):
        [x, y, theta] = self.get_self_stateGT()
        [ag_x, ag_y,ag_ang] = self.goal_roboPos
        local_x = (ag_x - x) * np.cos(theta) + (ag_y - y) * np.sin(theta)
        local_y = -(ag_x - x) * np.sin(theta) + (ag_y - y) * np.cos(theta)
        return [local_x, local_y]
    
    def get_goalRobo_speed(self):
        return self.speed_goalRobo

    def get_local_and_robo_pos(self):
        local_goal_pos = self.get_local_goal()
        local_robo_pos = self.get_local_goalRobo()
        return local_goal_pos+local_robo_pos
    
    def get_ad_and_ag_speed(self):
        ad_speed = self.get_self_speed()
        ag_speed = self.get_goalRobo_speed()
        return ad_speed+ag_speed




