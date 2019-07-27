import vrep
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import random
from queue import Queue
from rrt_path_planning.rrt_spline_based import RRTSpline
from rrt_path_planning.search_space import SearchSpace
from runnable.distance_metric import zedDistance


class Controller:
    def __init__(self, *args, **kwargs):
        # 各种参数
        self.clientID = kwargs['clientID']
        self.base_handle = kwargs['base_handle']
        self.vision_handles = []
        self.vision_handles.append(kwargs['vision_handle_0'])
        self.vision_handles.append(kwargs['vision_handle_1'])
        self.controller_handle = kwargs['controller_handle']
        self.synchronous = kwargs['synchronous']
        self.time_interval = kwargs['time_interval']
        self.name_handle_map = {
            'base': self.base_handle,
            'vision_0': self.vision_handles[0],
            'vision_1': self.vision_handles[1],
            'controller': self.controller_handle
        }
        self.v_max = kwargs['v_max']
        self.v_add = kwargs['v_add']
        self.v_sub = kwargs['v_sub']
        self.v_min = kwargs['v_min']
        self.v_constant = kwargs['v_constant']
        self.use_constant_v = kwargs['use_constant_v']
        self.v_now = 0
        self.left_time = 0
        
        # 内部用到的数据结构
        self.moving_queue = Queue()
        self.target_position = None
        self.target_orientation = None
        self.controller_position = None
        self.has_reset_cumul = False
        self.need_take_photo = False

    def moveTo(self, target_position, start_speed, end_speed, accurate_move):
        # start_speed: 0: start with 0, 1: start with v_now, 2: start with v_min
        # end_speed: 0: end with 0, 1: end with v_min
        # accurate_move: True: accurate move (may cause v change), False: not accurate move
        if not self.moving_queue.empty():
            self.moving_queue = Queue()
        self.target_position = np.array(target_position)
        now_position = self.getPosition('controller')
        self.controller_position = now_position
        delta_position = np.array(target_position)-np.array(now_position)
        distance = np.linalg.norm(delta_position, ord=2)
        if not self.use_constant_v:
            target_orientation = delta_position/distance
            v1 = 0 if start_speed == 0 else (self.v_now if start_speed == 1 else self.v_min)
            if self.target_orientation is not None and target_orientation.dot(self.target_orientation) <= 0.5:
                print("!", target_orientation.dot(self.target_orientation))
                v1 = 0
            v2 = 0 if end_speed == 0 else self.v_min
            self.target_orientation = target_orientation
            v_max_possible = math.sqrt((v2**2*self.v_add+v1**2*self.v_sub+2*distance*self.v_add*self.v_sub+v1*self.v_add*self.v_sub-v2*self.v_add*self.v_sub)/(self.v_add+self.v_sub))
            v_max = v_max_possible if v_max_possible <= self.v_max else self.v_max
            t_add = int((v_max-v1)/self.v_add)
            t_sub = int((v_max-v2)/self.v_sub)
            v_max = t_add*self.v_add+v1
            distance_constant = distance - t_add*(v_max+v1+self.v_add)/2 - t_sub*(v_max+v2-self.v_sub)/2
            t_constant = int(distance_constant/v_max)
            distance_remain = distance_constant - t_constant * v_max

            flag = not accurate_move
            v_now_tmp = v1
            for i in range(t_add):
                v_now_tmp = v_now_tmp + self.v_add
                self.moving_queue.put(v_now_tmp)
            for i in range(t_constant):
                self.moving_queue.put(v_now_tmp)
            for i in range(t_sub):
                v_now_tmp = v_now_tmp - self.v_sub
                if v_now_tmp <= distance_remain and not flag:
                    flag = True
                    self.moving_queue.put(distance_remain)
                self.moving_queue.put(v_now_tmp)
            if not flag:
                self.moving_queue.put(distance_remain)
            self.left_time = self.moving_queue.qsize()
        else:
            target_orientation = delta_position/distance

    def moveToWithSpeed(self, target_position, start_speed, end_speed, accurate_move, target_speed):
        # start_speed: 0: start with 0, 1: start with v_now, 2: start with v_min
        # end_speed: 0: end with 0, 1: end with v_min
        # accurate_move: True: accurate move (may cause v change), False: not accurate move
        if not self.moving_queue.empty():
            self.moving_queue = Queue()
        self.target_position = np.array(target_position)
        now_position = self.getPosition('controller')
        self.controller_position = now_position
        delta_position = np.array(target_position)-np.array(now_position)
        distance = np.linalg.norm(delta_position, ord=2)
        if not self.use_constant_v:
            target_orientation = delta_position/distance
            v1 = 0 if start_speed == 0 else (self.v_now if start_speed == 1 else self.v_min)
            if self.target_orientation is not None and target_orientation.dot(self.target_orientation) <= 0.5:
                print("!", target_orientation.dot(self.target_orientation))
                v1 = 0
            v2 = 0 if end_speed == 0 else self.v_min
            self.target_orientation = target_orientation
            v_max_possible = 1/(1/self.v_add+1/self.v_sub)*\
                (target_speed/self.v_add+target_speed/self.v_sub+1/(self.v_add*self.v_sub)*\
                math.sqrt((self.v_add+self.v_sub)*(target_speed**2*self.v_add-2*target_speed*v2*self.v_add+v2**2*self.v_add+target_speed**2*self.v_sub-2*target_speed*v1*self.v_sub+v1**2*self.v_sub+2*distance*self.v_add*self.v_sub+v1*self.v_add*self.v_sub-v2*self.v_add*self.v_sub)))
            v_max = v_max_possible if v_max_possible <= self.v_max else self.v_max
            t_add = int((v_max-v1)/self.v_add)
            t_sub = int((v_max-v2)/self.v_sub)
            v_max = t_add*self.v_add+v1
            distance_constant = distance - t_add*(v_max+v1+self.v_add)/2 - t_sub*(v_max+v2-self.v_sub)/2
            t_constant = int(distance_constant/(v_max-target_speed))
            distance_remain = distance_constant - t_constant * (v_max-target_speed)

            flag = not accurate_move
            v_now_tmp = v1
            for i in range(t_add):
                v_now_tmp = v_now_tmp + self.v_add
                self.moving_queue.put(v_now_tmp)
            for i in range(t_constant):
                self.moving_queue.put(v_now_tmp)
            for i in range(t_sub):
                v_now_tmp = v_now_tmp - self.v_sub
                if v_now_tmp <= distance_remain and not flag:
                    flag = True
                    self.moving_queue.put(distance_remain)
                self.moving_queue.put(v_now_tmp)
            if not flag:
                self.moving_queue.put(distance_remain)
            self.left_time = self.moving_queue.qsize()
        else:
            target_orientation = delta_position/distance

    def step_forward_move(self):
        result = {}
        if self.synchronous:
            if self.moving_queue.empty():
                result['flag'] = False
            else:
                speed = self.moving_queue.get()
                self.move(self.target_orientation, speed)
                self.left_time = self.moving_queue.qsize()
                # reset cumul if needed
                if self.has_reset_cumul:
                    vrep.simxSetIntegerSignal(self.clientID, 'clear', 1, vrep.simx_opmode_oneshot)
                    self.has_reset_cumul = False
                # take photos if needed
                if self.need_take_photo:
                    image_0 = self.getImage(0)
                    image_1 = self.getImage(1)
                    if image_0 is not None and image_1 is not None:
                        result['photos'] = [image_0, image_1]
                        self.need_take_photo = False
                result['flag'] = True
            return result
        else:
            assert 0

    def move(self, target_orientation, speed):
        self.v_now = speed
        speed = target_orientation * speed
        for i in range(3):
            self.controller_position[i] += speed[i]
        self.setPosition('controller', self.controller_position)
        vrep.simxSynchronousTrigger(self.clientID)

    # def take_photos_move(self, target_orientation, speed):
    #     self.photo_num_now += 1
    #     if self.photo_num_now == self.photo_interval:
    #         cv2.imwrite("images/"+str(self.image_number_now)+'zed0.jpg', self.getImage(0))
    #         zed_position = np.array(self.getPosition('vision_0'))
    #         zed_orientation = np.array(self.getOrientation('vision_0'))
    #         with open('images/result.txt', 'a') as f:
    #             f.write(str(self.image_number_now) + ' position: ' + str(zed_position) + ' orientation: ' + str(zed_orientation) + '\n')

    #         cv2.imwrite("images/"+str(self.image_number_now)+'zed1.jpg', self.getImage(1))
    #         zed_position = np.array(self.getPosition('vision_1'))
    #         zed_orientation = np.array(self.getOrientation('vision_1'))
    #         with open('images/result.txt', 'a') as f:
    #             f.write(str(self.image_number_now) + ' position: ' + str(zed_position) + ' orientation: ' + str(zed_orientation) + '\n')
    #         self.image_number_now += 1
    #         self.photo_num_now = 0

    def getImage(self, vision_handle_number):
        if self.synchronous:
            err, resolution, image = vrep.simxGetVisionSensorImage(self.clientID, self.vision_handles[vision_handle_number], 0, vrep.simx_opmode_blocking)
            if err != vrep.simx_return_ok:
                return None
            else:
                img = np.array(image,dtype=np.uint8)
                img.resize([resolution[1],resolution[0],3])
                return img
        else:
            assert 0

    def getOrientation(self, handle_name):
        handle = self.name_handle_map.get(handle_name)
        if handle is None:
            assert 0
        if self.synchronous:
            err, euler_angles = vrep.simxGetObjectOrientation(self.clientID, handle, -1, vrep.simx_opmode_blocking)
        else:
            assert 0
        return euler_angles

    def getPosition(self, handle_name):
        handle = self.name_handle_map.get(handle_name)
        if handle is None:
            assert 0
        if self.synchronous:
            err, position = vrep.simxGetObjectPosition(self.clientID, handle, -1, vrep.simx_opmode_blocking)
        else:
            assert 0
        return position
    
    def setPosition(self, handle_name, position):
        handle = self.name_handle_map.get(handle_name)
        if handle is None:
            assert 0
        if self.synchronous:
            vrep.simxSetObjectPosition(self.clientID, handle, -1, position, vrep.simx_opmode_blocking)
        else:
            vrep.simxSetObjectPosition(self.clientID, handle, -1, position, vrep.simx_opmode_streaming)

    def setRequiredParams(self):
        # required_params = {
        #     'pParam':6,
        #     'iParam':0.04,
        #     'dParam':0.08,
        #     'vParam':-2,
        #     'alphaE':0.3,
        #     'dAlphaE':1.1,  # 1.1
        #     'alphaCumul': 0,
        #     'betaE':0.6,
        #     'dBetaE':1.1, # 2.5
        #     'betaCumul':0.001,
        #     'sp2':0.15,
        #     'dsp2':4,
        #     'sp2Cumul':0.005,
        #     'sp1':0.08, #0.26
        #     'dsp1':1,  # 4.2
        #     'sp1Cumul':0,
        # }
        required_params = {
            'pParam':30,
            'iParam':0,
            'dParam':20,
            'vParam':-4,
            'alphaE':0.3,
            'dAlphaE':2,  # 1.1
            'alphaCumul': 0.01,
            'betaE':0.6,
            'dBetaE':3.5, # 2.5
            'betaCumul':0.02,
            'sp2':0.26,
            'dsp2':4.5,
            'sp2Cumul':0,
            'sp1':0.26, #0.26
            'dsp1':4.5,  # 4.2
            'sp1Cumul':0,
        }
        # key to edit: betaE, dBetaE, betaZCumul, sp1, dsp1
        for key, value in required_params.items():
            vrep.simxSetFloatSignal(self.clientID, key, value, vrep.simx_opmode_oneshot)

    def setParams(self, params):
        for key, value in params.items():
            vrep.simxSetFloatSignal(self.clientID, key, value, vrep.simx_opmode_oneshot)
        vrep.simxSetIntegerSignal(self.clientID, 'clear', 1, vrep.simx_opmode_oneshot)
        vrep.simxSetIntegerSignal(self.clientID, 'stop', 1, vrep.simx_opmode_oneshot)

    def clear_cumul(self):
        vrep.simxSetIntegerSignal(self.clientID, 'clear', 0, vrep.simx_opmode_oneshot)
        self.has_reset_cumul = True

    def to_take_photos(self):
        self.need_take_photo = True

    def stop(self):
        vrep.simxSetIntegerSignal(self.clientID, 'stop', 0, vrep.simx_opmode_oneshot)


def main():
    # finish first
    vrep.simxFinish(-1)

    # connect to server
    clientID = vrep.simxStart("127.0.0.1", 19997, True, True, 5000, 5)
    if clientID != -1:
        print("Connect Succesfully.")
    else:
        print("Connect failed.")

    # get handles
    _, vision_handle_0 = vrep.simxGetObjectHandle(clientID, "zed_vision0", vrep.simx_opmode_blocking)
    _, vision_handle_1 = vrep.simxGetObjectHandle(clientID, "zed_vision1", vrep.simx_opmode_blocking)
    _, controller_handle= vrep.simxGetObjectHandle(clientID, "Quadricopter_target", vrep.simx_opmode_blocking)
    _, base_handle = vrep.simxGetObjectHandle(clientID, "Quadricopter", vrep.simx_opmode_blocking)

    # set Controller
    synchronous_flag = True
    time_interval = 0.05
    flight_controller = Controller(
        clientID=clientID,
        base_handle=base_handle, 
        controller_handle=controller_handle, 
        vision_handle_0=vision_handle_0, 
        vision_handle_1=vision_handle_1,
        synchronous=synchronous_flag,
        time_interval=time_interval,
        v_max=0.05,
        v_add=0.0005,
        v_sub=0.0005,
        v_min=0.01,
        v_constant=0.02,
        use_constant_v=False,
        )

    # set required params
    flight_controller.setRequiredParams()

    # set controller position
    base_position = flight_controller.getPosition('base')
    flight_controller.setPosition('controller', base_position)

    # start simulation
    vrep.simxSynchronous(clientID, synchronous_flag)
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
    
    # api test
    flight_controller.moveTo(np.array([5, -9.6, 4]), 0, 0, True)
    flag = flight_controller.step_forward_move()
    print(flag)
    while flag:
        flag = flight_controller.step_forward_move()
    # image = flight_controller.getImage(0)
    # cv2.imwrite("image.jpg", image)
    # image_2 = flight_controller.getImage(1)
    # cv2.imwrite("image_2.jpg", image_2)
    # pos = zedDistance(clientID, image_2, image)
    # print(pos)
    # orientation = getOrientation(clientID, sensor_handle_1, synchronous_flag)
    # print(orientation)
    # orientation_2 = getOrientation(clientID, sensor_handle_2, synchronous_flag)
    # print(orientation_2)
    # position = getPosition(clientID, sensor_handle_1, synchronous_flag)
    # print(position)
    # position_2 = getPosition(clientID, sensor_handle_2, synchronous_flag)
    # print(position_2)

    # take photos
    # target_position = np.array(base_position) + np.array([0, 0, 2])
    # flight_controller.moveTo(target_position)

    # target_position = target_position + np.array([1, 1, 0])
    # flight_controller.moveTo(target_position)
    # target_position = target_position + np.array([1, -1, 0])
    # flight_controller.moveTo(target_position)
    # target_position = target_position + np.array([-1, -1, 0])
    # flight_controller.moveTo(target_position)
    # target_position = target_position + np.array([-1, 1, 0])
    # flight_controller.moveTo(target_position)
    
    # target_position = target_position + np.array([0, 0, -2])
    # flight_controller.moveTo(target_position)

    # 场景大小
    x_coordinate = (-50, 50)
    y_coordinate = (-50, 50)

    # 初始位置：base_position
    # 终止位置：base_position + [3, 3]
    # base_position_sendin = (base_position[0]*10, base_position[1]*10)
    # target_position_sendin = (base_position_sendin[0]+50, base_position_sendin[1])
    # search_space = SearchSpace(np.array([x_coordinate, y_coordinate]))
    # print(base_position)
    
    # print(base_position_sendin)
    # rrt_spline = RRTSpline(search_space, base_position_sendin, target_position_sendin)
    # path = rrt_spline.rrt_search()
    # print(path)
    # move_count = 0
    # for position in path[0][1:]:
    #     print("position:", position)
    #     target_position = np.append(np.array(position)/10, base_position[2])
    #     print("target_position:", target_position)
    #     print(move_count)
    #     if move_count == 0:
    #         if move_count == len(path[0])-2:
    #             print("first and final")
    #             flight_controller.moveTo(target_position, type=3)
    #         else:
    #             print("first")
    #             flight_controller.moveTo(target_position, type=1)
    #     elif move_count == len(path[0])-2:
    #         print("final")
    #         flight_controller.moveTo(target_position, type=2)
    #     else:
    #         flight_controller.moveTo(target_position, type=0)
    #     move_count += 1

    # for i in range(50):
    #     flight_controller.step_forward()
    # target_position_now = np.array(flight_controller.getPosition('controller'))
    # print(target_position-target_position_now)

    # stop simulation
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)

    # finish
    vrep.simxFinish(clientID)


if __name__ == '__main__':
    main()