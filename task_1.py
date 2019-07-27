from apis_2 import Controller
import vrep
import cv2
import numpy as np
from runnable.distance_metric import zedDistance


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

    start_position = [-1, -2.725, 5]
    target_height = 0.495
    # 发现目标
    while True:
        flight_controller.moveTo(np.array([start_position[0], start_position[1], target_height]), 0, 0, True)
        flight_controller.to_take_photos()
        while True:
            result = flight_controller.step_forward_move()
            if 'photos' in result:
                print("Get Photos")
                pos = zedDistance(clientID, result['photos'][1], result['photos'][0])
                print(pos)
                break
            else:
                print("not get photos")
    
    # stop simulation
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)

    # finish
    vrep.simxFinish(clientID)


if __name__ == '__main__':
    main()
    