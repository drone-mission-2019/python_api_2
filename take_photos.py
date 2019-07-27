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

    # move to start position
    height = 4
    count = 0
    target_x = 0.45
    target_y = -2.5
    # move_points = [[7, -4.9, height], [4.875, -6.775, height], [1.225, -6.5, height], [-1.375, -3.275, height], [0.65, 0.6, height], [4, 1.3, height]]
    move_points = [[base_position[0], base_position[1], 3], [base_position[0], base_position[1], 4]]
    photo_intervals = 20
    photo_num = 0
    photo_count_now = 0
    pos = None
    while True:
        for i in range(len(move_points)):
            print("move_points:", move_points)
            if count == 0:
                flight_controller.moveTo(np.array(move_points[i]), 0, 1, True)
            else:
                flight_controller.moveTo(np.array(move_points[i]), 1, 1, False)
            flight_controller.clear_cumul()
            result = flight_controller.step_forward_move()
            while result['flag']:
                photo_count_now += 1
                if photo_count_now == photo_intervals:
                    photo_num += 1
                    flight_controller.to_take_photos()
                    photo_count_now = 0
                if 'photos' in result.keys():
                    print("Get Photos")
                    cv2.imwrite("images/"+ str(photo_num) + "zed0.jpg", result['photos'][0])
                    cv2.imwrite("images/"+ str(photo_num) + "zed1.jpg", result['photos'][1])
                    zed_position_0 = np.array(flight_controller.getPosition('vision_0'))
                    zed_orientation_0 = np.array(flight_controller.getOrientation('vision_0'))
                    zed_position_1 = np.array(flight_controller.getPosition('vision_1'))
                    zed_orientation_1 = np.array(flight_controller.getOrientation('vision_1'))
                    with open('images/result.txt', 'a') as f:
                        f.write('photo num: ' + str(photo_num) + ' zed0 position: ' + str(zed_position_0) + ' orientation: ' + str(zed_orientation_0) + ' zed1 position: ' + str(zed_position_1) + ' orientation: ' + str(zed_orientation_1) + '\n')
                    pos_new = zedDistance(clientID, result['photos'][1], result['photos'][0])
                    if pos_new is not None and (pos is None or (pos_new[0] > pos[0] and pos_new[0]-pos[0]<0.6)):
                        speed = pos_now - pos
                        pos = pos_new
                        pos[2] = 0.5
                        flight_controller.moveTo(np.array(pos), 1, 1, True)
                        target_x = pos[0]
                        target_y = pos[1]
                        move_points = [[target_x, target_y, 3], [target_x, target_y, 2]]
                result = flight_controller.step_forward_move()
            count += 1
    
    # stop simulation
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)

    # finish
    vrep.simxFinish(clientID)


if __name__ == '__main__':
    main()