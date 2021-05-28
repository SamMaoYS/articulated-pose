from os import read
import os
from posixpath import join
import cv2
import json
import numpy as np
import yaml
import h5py
import open3d as o3d
import pdb


def get_arrow(origin=[0, 0, 0], end=None, color=[0, 0, 0]):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    # origin[0], origin[1], origin[2] = origin[1], origin[2], origin[0]
    # end[0], end[1], end[2] = end[1], end[2], end[0]

    # trans = np.array([[  0.0000000,  0.0000000, -1.0000000],
    # [-1.0000000,  0.0000000, -0.0000000],
    # [0.0000000,  1.0000000,  0.0000000]])

    # trans2 = np.array([[-1.0000000,  0.0000000,  0.0000000],
    # [0.0000000,  1.0000000,  0.0000000],
    # [-0.0000000,  0.0000000, -1.0000000]])

    # trans = np.dot(trans, trans2)
    # origin = np.dot(origin, trans)
    # end = np.dot(end, trans)

    vec_Arr = np.array(end) - np.array(origin)
    vec_len = np.linalg.norm(vec_Arr)

    mesh_arrow = o3d.geometry.create_mesh_arrow(
        cone_height=0.2 * vec_len,
        cone_radius=0.06,
        cylinder_height=0.8 * vec_len,
        cylinder_radius=0.04,
    )
    mesh_arrow.paint_uniform_color(color)
    rot_mat = caculate_align_mat(vec_Arr)

    transorm_mat = np.identity(4)
    transorm_mat[:3, :3] = rot_mat
    transorm_mat[:3, 3] = origin
    mesh_arrow.transform(transorm_mat)

    return mesh_arrow


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)
    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:   
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                        z_c_vec_mat)/(1 + np.dot(z_unit_Arr, pVec_Arr))
    qTrans_Mat *= scale
    return qTrans_Mat

def read_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    filename = "predictions.json"
    dataset_path = "dataset/sapien/render/drawer"
    predictions = read_json(filename)

    max_x, max_y = 1000, 10000
    for key, joint in predictions.items():
        joint_pt = np.asarray(joint["joint_pt"][0])
        joint_axis = np.asarray(joint["joint_axis"][0])
        # joint_axis /= np.linalg.norm(joint_axis)
        print(joint_pt)
        img_id, pose_id, view_id = key.split('_')
        basepath = os.path.join(dataset_path, img_id, pose_id)
        img_path = os.path.join(basepath, "rgb", view_id+".png")
        yaml_path = os.path.join(basepath, "gt.yml")

        img = cv2.imread(img_path)
        depth = np.array(h5py.File(os.path.join(basepath, "depth", view_id+".h5"), "r")["data"])
        with open(yaml_path, "r") as f:
            meta_instance = yaml.load(f, Loader=yaml.Loader)

        # view_mat = np.asarray(meta_instance["frame_"+view_id]['viewMat']).reshape((4,4)).transpose()
        view_mat = np.asarray(joint["transform"])
        projection = np.asarray(meta_instance["frame_"+view_id]['projMat']).reshape((4,4)).transpose()
        # origin_2d = np.dot(projection, np.concatenate([joint_pt, [1.0]]))
        # origin_2d /= origin_2d[-1]
        # origin_2d[0], origin_2d[1] = (origin_2d[0]+1.0)/2*img.shape[1], (origin_2d[1]+1.0)/2*img.shape[0]
        # origin_2d = origin_2d[:2].astype(int)

        # end_2d = np.dot(projection, np.concatenate([joint_pt+joint_axis, [1.0]]))
        # end_2d /= end_2d[-1]
        # end_2d[0], end_2d[1] = (end_2d[0]+1.0)/2*img.shape[1], (end_2d[1]+1.0)/2*img.shape[0]
        # end_2d = end_2d[:2].astype(int)
        # print(projection)
        mask = depth > 0
        nx, ny = (img.shape[1], img.shape[0])
        x = np.linspace(-1, 1, nx)
        y = np.linspace(1, -1, ny)
        y1 = np.linspace(-1, 1, ny)
        xv, yv = np.meshgrid(x, y)
        xv1, yv1 = np.meshgrid(x, y1)
        # xv = xv[mask]
        # yv = yv[mask]
        # depth = depth[mask]*1000
        # img_3d = np.stack([xv, yv, depth, np.ones_like(depth)], axis=1).reshape((-1, 4)).transpose()

        w_channel = -depth
        projected_map = np.stack(
            [xv * w_channel, yv * w_channel, depth, w_channel]
        ).transpose([1, 2, 0])
        projected_map1 = np.stack(
            [xv * w_channel, yv1 * w_channel, depth, w_channel]
        ).transpose([1, 2, 0])

        projected_points = projected_map[mask]
        depth_channel = -projected_points[:, 3:4]
        cloud_cam = np.dot(
            projected_points[:, 0:2] - np.dot(depth_channel, projection[0:2, 2:3].T),
            np.linalg.pinv(projection[:2, :2].T),
        )

        projected_points1 = projected_map1[mask]
        projected_points1 = np.reshape(projected_points1, [-1, 4])
        cloud_cam_real = np.dot(
            projected_points1[:, 0:2] - np.dot(depth_channel, projection[0:2, 2:3].T),
            np.linalg.pinv(projection[:2, :2].T),
        )
        cloud_cam_real = np.concatenate((cloud_cam_real, depth_channel), axis=1)

        cloud_cam = np.concatenate((cloud_cam, depth_channel), axis=1)
        cloud_cam_full = np.concatenate(
            (cloud_cam, np.ones((cloud_cam.shape[0], 1))), axis=1
        )

        # modify, todo
        camera_pose_mat = np.linalg.pinv(view_mat.T)
        camera_pose_mat[:3, :] = -camera_pose_mat[:3, :]
        cloud_world = np.dot(cloud_cam_full, camera_pose_mat)

        print(cloud_cam_full.shape)
        
        # cam_3d = np.dot(np.linalg.inv(projection), img_3d)
        # cam_3d /= cam_3d[-1, :]
        # world_3d = np.dot(np.linalg.inv(view_mat), cam_3d)
        # world_3d /= world_3d[-1, :]

        arrow = get_arrow(joint_pt+joint_axis, joint_pt)

        convention = np.asarray([[-1, 0, 0], [0,1, 0], [0,0,-1]])
        trans = np.asarray([[1, 0, 0], [0,0,-1], [0,1,0]])
        trans = np.dot(convention, np.linalg.inv(trans))

        o3d_vertices = o3d.utility.Vector3dVector(cloud_cam_real[:, :3])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d_vertices
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f'{key}', width=640, height=640, visible=True)
        vis.add_geometry(pcd)
        vis.add_geometry(o3d.geometry.create_mesh_coordinate_frame())
        vis.add_geometry(arrow)
        vis.run()
        vis.destroy_window()
        del vis
        

        # print(origin_2d)
        # cv2.circle(img, (origin_2d[0], origin_2d[1]), 5, (0,0,255), 2)
        # cv2.circle(img, (end_2d[0], end_2d[1]), 5, (255,0,0), 2)
        # cv2.arrowedLine(img, (origin_2d[0], origin_2d[1]), (end_2d[0], end_2d[1]), (1,0,0), 2)
        # cv2.imshow(f"key", img)
        # cv2.waitKey(0)
        




