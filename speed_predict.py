import numpy as np
import math


class SpeedPredictor:
    def __init__(self, target_height, bounds, max_predict_step=5, extra_time=0.1):
        self.time = 0
        self.position_list = []
        self.time_list = []
        self.target_height = target_height
        self.bounds = bounds
        self.max_predict_step = max_predict_step
        self.extra_time = extra_time
        self.max_left_time = 1000
        self.max_delta_position = 1

    def give_new_information(self, position):
        if position[0] < self.bounds[0][0] or position[0] > self.bounds[0][1] or position[1] < self.bounds[1][0] or position[1] > self.bounds[1][1]:
            return
        self.position_list.append(position)
        self.time_list.append(self.time)

    def step_forward(self):
        self.time += 1

    def get_next_target(self, controller_position, left_time):
        left_time = int(left_time * (1+self.extra_time))
        if left_time > self.max_left_time:
            return None
        if len(self.position_list) == 1:
            return None
        delta_distance_list = []
        delta_time_list = []
        delta_step = len(self.position_list)//2
        if delta_step > self.max_predict_step:
            for i in range(self.max_predict_step):
                delta_distance_list.append(self.position_list[len(self.position_list)-self.max_predict_step+i][:2] - self.position_list[len(self.position_list)-2*self.max_predict_step+i][:2])
                delta_time_list.append(self.time_list[len(self.position_list)-self.max_predict_step+i] - self.time_list[len(self.position_list)-2*self.max_predict_step+i])
        else:
            for i in range(delta_step):
                delta_distance_list.append(self.position_list[i+delta_step][:2] - self.position_list[i][:2])
                delta_time_list.append(self.time_list[i+delta_step]-self.time_list[i])
        total_delta_time = sum(delta_time_list)
        total_delta_distance = sum(delta_distance_list)
        if np.linalg.norm(total_delta_distance) == 0:
            target_orientation = [0,0]
        else:
            target_orientation = total_delta_distance/np.linalg.norm(total_delta_distance)
        target_speed = np.linalg.norm(total_delta_distance)/total_delta_time
        final_position = left_time * target_orientation * target_speed + self.position_list[-1][:2]
        final_position = list(final_position)
        delta_position = np.array(controller_position[:2])-self.position_list[-1][:2]
        if np.linalg.norm(delta_position)>=self.max_delta_position:
            final_position.append(controller_position[2])
        else:
            final_position.append(self.target_height)
        final_position = np.array(final_position)
        return final_position
