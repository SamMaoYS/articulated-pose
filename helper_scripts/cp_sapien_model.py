import os

sapien_dir = "/home/hja40/Desktop/Dataset/data/models3d/partnetsim/mobility_v1_alpha5/"
output_dir = "/home/hja40/Desktop/Research/articulated-pose/dataset/sapien/urdf/drawer/"

def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == "__main__":
    # Copy drawer models from mobility_v1_alpha5 into sapien
    train_list=['40453', '44962', '45132',
                    '45290', '46130', '46334',  '46462',
                    '46537', '46544', '46641', '47178', '47183',
                    '47296', '47233', '48010', '48253',  '48517',
                    '48740', '48876', '46230', '44853', '45135',
                    '45427', '45756', '46653', '46879', '47438', '47711', '48491']
    test_list=[ '46123',  '45841', '46440']
    models = train_list + test_list

    existDir(output_dir)

    for model in models:
        existDir(output_dir + model)
        os.system(f'cp {sapien_dir + model + "/mobility.urdf"} {output_dir + model + "/"}')
        os.system(f'cp -r {sapien_dir + model + "/textured_objs"} {output_dir + model + "/"}')
        os.system(f'cp -r {sapien_dir + model + "/images"} {output_dir + model + "/"}')
    