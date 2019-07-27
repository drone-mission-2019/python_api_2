from apis_2 import Controller
import vrep
import cv2
import numpy as np
from runnable.distance_metric import zedDistance
from speed_predict import SpeedPredictor


def main():
    # finish first
    vrep.simxFinish(-1)

    # connect to server
    clientID = vrep.simxStart("127.0.0.1", 19997, True, True, 5000, 5)
    if clientID != -1:
        print("Connect Succesfully.")
    else:
        print("Connect failed.")
        assert 0
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)

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
        v_max=0.03,
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
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxSetIntegerSignal(clientID, 'stop', 1, vrep.simx_opmode_oneshot)
    vrep.simxSynchronousTrigger(clientID)

    target_height = 0.48
    bounds = [[-5, 5], [-5, 5]]

    while True:
        # 巡航搜索
        search_points = [[-2.5, -2.5], [-2.5, 0], [-2.5, 2.5], [0, 2.5], [2.5, 2.5], [2.5, 0], [2.5, -2.5], [0, -2.5]]
        photo_interval = 20
        photo_count = photo_interval
        find_flag = False
        while True:
            for target_point in search_points:
                target_point.append(base_position[2])
                target_point = np.array(target_point)
                flight_controller.moveTo(target_point, 1, 1, True)
                while np.linalg.norm(flight_controller.controller_position - target_point) >= 0.01:
                    if photo_count >= photo_interval:
                        flight_controller.to_take_photos()
                        photo_count = 0
                    else:
                        photo_count += 1
                    result = flight_controller.step_forward_move()
                    if 'photos' in result:
                        pos = zedDistance(clientID, result['photos'][1], result['photos'][0])
                        if pos is not None:
                            find_flag = True
                            target_position = np.array([pos[0], pos[1], base_position[2]])
                            break
                if find_flag:
                    break
            if find_flag:
                break
        print("Find Target", target_position)
                            
        # 发现目标
        speed_predictor = SpeedPredictor(target_height, bounds, max_predict_step=1, extra_time=0)
        photo_interval = 5
        photo_count = photo_interval
        stop_height = 1.3
        flag = True
        while True:
            flight_controller.moveTo(target_position, 1, 1, True)
            print(target_position)
            if photo_count >= photo_interval and flight_controller.getPosition('controller')[2] >= stop_height:
                flight_controller.to_take_photos()
                photo_count = 0
            else:
                photo_count += 1
            result = flight_controller.step_forward_move()
            speed_predictor.step_forward()
            if 'photos' in result:
                print("Get Photos")
                pos = zedDistance(clientID, result['photos'][1], result['photos'][0])
                # _, land_handle = vrep.simxGetObjectHandle(clientID, "land_plane", vrep.simx_opmode_blocking)
                # _, actual_pos = vrep.simxGetObjectPosition(clientID, land_handle, -1, vrep.simx_opmode_blocking)
                # pos = actual_pos
                if pos is not None:
                    speed_predictor.give_new_information(np.array(pos))
                new_target_position = speed_predictor.get_next_target(flight_controller.controller_position, flight_controller.left_time)
                if new_target_position is not None:
                    left_time_before = flight_controller.left_time
                    flight_controller.moveTo(new_target_position, 1, 1, True)
                    flag = True  # 防止不收敛情况出现
                    count = 0
                    while flight_controller.left_time != left_time_before :
                        if count >= 10:
                            flag = False
                            break
                        new_target_position = speed_predictor.get_next_target(flight_controller.controller_position, flight_controller.left_time)
                        if new_target_position is None:
                            flag = False
                            break
                        left_time_before = flight_controller.left_time
                        flight_controller.moveTo(new_target_position, 1, 1, True)
                        count += 1
                    if flag:
                        target_position = new_target_position
            if flight_controller.getPosition('controller')[2] <= 0.5:
                break
    
    # stop simulation
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)

    # finish
    vrep.simxFinish(clientID)


def examine(pos, actual_pos, image_0=None, image_1=None, zed_0_orientation=None, zed_1_orientation=None):
    pos = np.array(pos)
    actual_pos = np.array(actual_pos)
    print(pos, actual_pos)
    if (pos[0]-actual_pos[0])**2 >= 0.01:
        print(pos, actual_pos)
    


if __name__ == '__main__':
    main()
    