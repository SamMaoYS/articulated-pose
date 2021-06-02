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

def fix_coordinate():
    trans = np.array([[  0.0000000,  0.0000000, -1.0000000],
    [-1.0000000,  0.0000000, -0.0000000],
    [0.0000000,  1.0000000,  0.0000000]])

    trans2 = np.array([[-1.0000000,  0.0000000,  0.0000000],
    [0.0000000,  1.0000000,  0.0000000],
    [-0.0000000,  0.0000000, -1.0000000]])

    trans = np.dot(trans, trans2)
    return trans

def get_arrow(origin=[0, 0, 0], end=None, color=[0, 0, 0]):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """

    # trans = fix_coordinate()

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
        joint_pt = np.asarray(joint["joint_pt"])
        joint_axis = np.asarray(joint["joint_axis"])

        joint_pt_gt = np.asarray(joint["joint_pt_gt"])
        joint_axis_gt = np.asarray(joint["joint_axis_gt"])

        print(key)
        
        img_id, pose_id, view_id = key.split('_')
        basepath = os.path.join(dataset_path, img_id, pose_id)
        img_path = os.path.join(basepath, "rgb", view_id+".png")
        yaml_path = os.path.join(basepath, "gt.yml")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        depth = np.array(h5py.File(os.path.join(basepath, "depth", view_id+".h5"), "r")["data"])
        with open(yaml_path, "r") as f:
            meta_instance = yaml.load(f, Loader=yaml.Loader)

        view_mat = np.asarray(meta_instance["frame_"+view_id]['viewMat']).reshape((4,4)).transpose()
        jt_rotation = np.asarray(joint["rotation"])
        jt_translation = np.asarray(joint["translation"])
        jt_scale = np.asarray(joint["scale"])
        jt_transform = np.identity(4)
        jt_transform[:3, :3] = jt_rotation
        jt_transform[:3, 3] = jt_translation
        jt_transform[:3, :3] *= jt_scale
        projection = np.asarray(meta_instance["frame_"+view_id]['projMat']).reshape((4,4)).transpose()

        mask = depth > 0
        nx, ny = (img.shape[1], img.shape[0])
        x = np.linspace(-1, 1, nx)
        y = np.linspace(1, -1, ny)
        y1 = np.linspace(-1, 1, ny)
        xv, yv = np.meshgrid(x, y)
        xv1, yv1 = np.meshgrid(x, y1)
        xv = xv[mask]
        yv = yv[mask]
        depth = depth[mask]

        z_buffer = -depth * projection[2,2] + projection[2,3]
        w_buffer = -depth * projection[3,2] + projection[3,3]
        z_buffer /= w_buffer
        img_3d = np.stack([xv, yv, z_buffer, np.ones_like(z_buffer)], axis=1).reshape((-1, 4)).transpose()
        
        cam_3d = np.dot(np.linalg.inv(projection), img_3d)
        cam_3d /= cam_3d[-1, :]
        world_3d = np.dot(np.linalg.inv(view_mat), cam_3d)
        world_3d /= world_3d[-1, :]
        cam_3d = cam_3d.transpose()
        world_3d = world_3d.transpose()

        joint_axis_2 = joint_axis / np.linalg.norm(joint_axis)
        joint_axis_gt_2 = joint_axis_gt / np.linalg.norm(joint_axis_gt)
        angle = np.abs(np.dot(joint_axis_2, joint_axis_gt_2))
        if angle > 1.0:
            angle = 1.0
        angle = np.arccos(angle) * 180/np.pi
        
        _COLORS_LEVEL = {0: np.array([0, 255, 0])/255, 1: np.array([255, 128, 0])/255, 2: np.array([255, 0, 0])/255}
        if angle < 2.0:
            color = _COLORS_LEVEL[0]
        elif angle >= 2.0 and angle < 10:
            color = _COLORS_LEVEL[1]
        elif angle >= 10.0:
            color = _COLORS_LEVEL[2]
        
        joint_transform = np.linalg.inv(jt_transform)
        joint_pt = np.dot(joint_transform, np.concatenate([joint_pt, [1]]))
        joint_axis = joint_axis / np.linalg.norm(joint_axis)
        joint_axis = np.dot(joint_transform[:3, :3], joint_axis)
        joint_pt = joint_pt[:3]
        end_pt = joint_pt + joint_axis

        trans = fix_coordinate()
        joint_pt = np.dot(joint_pt, trans)
        end_pt = np.dot(end_pt, trans)

        joint_pt = np.dot(view_mat, np.concatenate([joint_pt, [1]]))
        joint_pt = joint_pt[:3]
        end_pt = np.dot(view_mat, np.concatenate([end_pt, [1]]))
        end_pt = end_pt[:3]

        arrow = get_arrow(joint_pt, end_pt, color)

        o3d_vertices = o3d.utility.Vector3dVector(cam_3d[:, :3])

        colors = img[mask]
        o3d_colors = o3d.utility.Vector3dVector(colors/255.0)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d_vertices
        # pcd.colors = o3d_colors
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(window_name=f'{key}', width=640, height=640, visible=True)
        # opt = vis.get_render_option()
        # opt.background_color = np.asarray([0.5, 0.5, 0.5])
        # vis.add_geometry(pcd)
        # # vis.add_geometry(o3d.geometry.create_mesh_coordinate_frame())
        # vis.add_geometry(arrow)
        # vis.run()
        # vis.destroy_window()
        # del vis
        # del opt

        ndc_joint_pt = np.dot(projection, np.concatenate([joint_pt, [1]]))
        ndc_joint_pt = ndc_joint_pt / ndc_joint_pt[-1]

        ndc_end_pt = np.dot(projection, np.concatenate([end_pt, [1]]))
        ndc_end_pt = ndc_end_pt / ndc_end_pt[-1]
        
        ndc_joint_pt[1] = -ndc_joint_pt[1]
        sx, sy = ((ndc_joint_pt[:2]+1.0)/2.0 * img.shape[0]).astype(int)
        ndc_end_pt[1] = -ndc_end_pt[1]
        ex, ey = ((ndc_end_pt[:2]+1.0)/2.0 * img.shape[0]).astype(int)

        print(sx, sy)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.circle(img, (sx, sy), 5, (0,0,255), 2)
        cv2.circle(img, (ex, ey), 5, (255,0,0), 2)
        cv2.arrowedLine(img, (sx, sy), (ex, ey), (255,0,0), 2)
        cv2.imshow(f"key", img)
        cv2.waitKey(0)
        




