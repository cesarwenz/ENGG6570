from data_loader import *

# load target dataset
loader = DataLoader(base_path='D:\\Data', dataset_name='0004')

obs_len = 10
future_len = 30
seq_len = loader.data.shape[0]
start_idx = 10


for i in range(start_idx, seq_len - future_len - 1):
    print('>> current frm idx %d' % i)

    # calibration params
    K = loader.K
    Rt = loader.Rt
    '''
    ** data format **135,.2dfrtuz
    data[:, 0] : frame index 
    data[:, 1] : altitude [meter]
    data[:, 2] : roll [degree]
    data[:, 3] : pitch [degree]
    data[:, 4] : yaw [degree]
    data[:, 5] : heading [degree]
    data[:, 6] : TM X-axis [meter]
    data[:, 7] : TM Y-axis [meter]
    data[:, 8] : img file name
    data[:, 9] : point cloud file name
    data[:, 10] : steering angle [degree]
    data[:, 11] : speed [km/h]
    data[:, 12] : turn signal, non (0), left (1), right (2), both (3)
    data[:, 13] : gps localization error std [meter]
    data[:, 14] : driving action
    '''

    # current frame data
    cur_data = loader.data[i-obs_len: i+future_len]
    
    drv_act = cur_data[obs_len, 14]

    # transformation matrix (global to egocentric)
    Mt = loader.create_pose_matrix(cur_data[obs_len][1], cur_data[obs_len][2], cur_data[obs_len][3],
                                   cur_data[obs_len][4], cur_data[obs_len][6], cur_data[obs_len][7])

    # global trajectory in homo
    global_traj_xy = cur_data[:, 6:8]
    global_traj_alt = cur_data[:, 1].reshape(global_traj_xy.shape[0], 1) - 0.35
    global_traj_homo = np.concatenate([global_traj_xy, global_traj_alt, np.ones(shape=(global_traj_xy.shape[0], 1))], axis=1)

    # ego-centric traj
    ego_traj = np.matmul(Mt, global_traj_homo.T).T

    # draw trajectory on left image and write driving action on the imaged

    
    # image = resize_image(img, 0.5, 0.88889)
    img = loader.draw_traj_on_left_img(ego_traj, cur_data[obs_len, 8], obs_len, drv_act)
    print(ego_traj.shape[0])
    cv2.imshow('test', img)
    cv2.waitKey(0)

    # Save Image
    # directory = r'D:\Data\0004\labelled'
    # os.chdir(directory)
    # filename = ('savedImage%d.jpg' % i)
    # cv2.imwrite(filename, img)
 