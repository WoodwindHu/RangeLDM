
import pdb
import cv2
import numpy as np
import torch
import glob
import os



def range_image_to_point_cloud_fast_with_intensity(range_image):
    depth_range = range_image[0]

    fov_up=3.0
    fov_down=-25.0
    fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    W = 1024.0
    H = 64.0
    x, y = np.meshgrid(np.arange(0, W), np.arange(0, H))

    x *= 1/W
    y *= 1/H

    yaw = np.pi*(x * 2 - 1)
    pitch = (1.0 - y)*fov - abs(fov_down)

    yaw = yaw.flatten()
    pitch = pitch.flatten()
    depth = depth_range.flatten()
    intensity = range_image[1].flatten()

    pts = np.zeros((len(yaw), 4), dtype=np.float32)
    pts[:, 0] =  np.cos(yaw) * np.cos(pitch) * depth
    pts[:, 1] =  -np.sin(yaw) * np.cos(pitch) * depth
    pts[:, 2] =  np.sin(pitch) * depth
    pts[:, 3] = intensity

    mask = np.logical_and(depth>3.0, depth < 70.0)
    xyzi = pts[mask, :]
    return xyzi

def calculate_mae(exp_dir):
    error_bc = 0.0
    error_nn = 0.0
    error_ours = 0.0

    sample_count = len(glob.glob(exp_dir + '/densification_target/*.pth'))
    os.makedirs(exp_dir + '/densification_bicubic', exist_ok=True)
    os.makedirs(exp_dir + '/densification_nn', exist_ok=True)
    for idx in range(sample_count):
        result_path = (exp_dir + '/densification_result/{0}.pth'.format(str(idx)))
        target_path = (exp_dir + '/densification_target/{0}.pth'.format(str(idx)))

        result = torch.load(result_path).cpu().numpy()[0]
        target = torch.load(target_path).cpu().numpy()[0]

        result = np.exp2(result*6)-1
        target = np.exp2(target*6)-1

        result_bc = target[::4]
        result_bc_i = torch.load(target_path).cpu().numpy()[1][::4]
        result_bc = cv2.resize(result_bc, (0, 0), fx=1.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
        result_bc_i = cv2.resize(result_bc_i, (0, 0), fx=1.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
        bc = np.stack((result_bc, result_bc_i), axis=0).astype(np.float32)
        bc = range_image_to_point_cloud_fast_with_intensity(bc)
        bc.tofile(exp_dir + '/densification_bicubic/{0}.bin'.format(str(idx)))

        result_nn = target[::4]
        result_nn_i = torch.load(target_path).cpu().numpy()[1][::4]
        result_nn = cv2.resize(result_nn, (0, 0), fx=1.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
        result_nn_i = cv2.resize(result_nn_i, (0, 0), fx=1.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
        nn = np.stack((result_nn, result_nn_i), axis=0).astype(np.float32)
        nn = range_image_to_point_cloud_fast_with_intensity(nn)
        nn.tofile(exp_dir + '/densification_nn/{0}.bin'.format(str(idx)))

        error_bc += np.sum(np.abs(result_bc - target))
        error_nn += np.sum(np.abs(result_nn - target))
        error_ours += np.sum(np.abs(result - target))

    count = sample_count * 1024 * 64

    error_bc = error_bc / count
    error_nn = error_nn / count
    error_ours = error_ours / count

    print('-------------------------------------------------')
    print('MAE Bicubic:  ' + str(error_bc))
    print('MAE NN:       ' + str(error_nn))
    print('MAE Ours: ' + str(error_ours))
    print('-------------------------------------------------')

def calculate_inpainting_mae(exp_dir):
    error_ours = 0.0

    sample_count = len(glob.glob(exp_dir + '/inpainting_target/*.pth'))
    for idx in range(sample_count):
        result_path = (exp_dir + '/inpainting_result/{0}.pth'.format(str(idx)))
        target_path = (exp_dir + '/inpainting_target/{0}.pth'.format(str(idx)))

        result = torch.load(result_path).cpu().numpy()[0][:,:64]
        target = torch.load(target_path).cpu().numpy()[0][:,:64]

        result = np.exp2(result*6)-1
        target = np.exp2(target*6)-1

        error_ours += np.sum(np.abs(result - target))

    count = sample_count * 1024 * 64

    error_ours = error_ours / count

    print('-------------------------------------------------')
    print('MAE Ours: ' + str(error_ours))
    print('-------------------------------------------------')

def bin2range(filename):
    real, intensity = point_cloud_to_range_image(filename, False, True)
    #Make negatives 0
    real = np.where(real<0, 0, real) + 0.0001
    #Apply log
    real = ((np.log2(real+1)) / 6)
    #Make negatives 0
    real = np.clip(real, 0, 1)
    real = np.expand_dims(real, axis = 0)
    intensity = np.clip(intensity, 0, 1.0)
    intensity = np.expand_dims(intensity, axis = 0)
    real = np.concatenate((real, intensity), axis = 0)
    real = torch.from_numpy(real)
    return real        
    