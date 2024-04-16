import trimesh
import pyrender
import numpy as np
from PIL import Image
import os.path as osp
import json
import glob
import cv2
import os
import PIL
import shutil
import pickle
from plyfile import PlyData

"""Step 1: 生成NOCS map"""
"生成NOCS model"
fuze_trimesh = trimesh.load('C:/Users/24762/Desktop/bop/obj_000000.ply') #导入模型文件
# 更新fuze_trimesh
# 遍历物体的所有顶点
for i in range(len(fuze_trimesh.vertices)):
    # 将顶点坐标归一化到[-0.5, 0.5]的范围内
    normalized_vertex = (fuze_trimesh.vertices[i] - fuze_trimesh.bounds[0]) / (
                fuze_trimesh.bounds[1] - fuze_trimesh.bounds[0]) - 0.5

    # 将归一化后的顶点坐标映射到[0,1]之间，作为顶点的颜色
    # color = (normalized_vertex + 0.5) % 1
    color = 255 * (normalized_vertex + 0.5)

    # 增加透明度
    alpha = 255

    new_color = np.append(color, alpha)

    # 更新顶点颜色
    fuze_trimesh.visual.vertex_colors[i] = new_color

# 保存新的model为new_model.ply
fuze_trimesh.export('C:/Users/24762/Desktop/bop/obj_000000_nocs.ply')  # 保存NOCS_model

# mesh = pyrender.Mesh.from_trimesh(trimesh.load('./new_model.ply'))
mesh = pyrender.Mesh.from_trimesh(trimesh.load(fuze_trimesh))
scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)

"生成NOCS map"
# 加载物体模型
mesh = trimesh.load('C:/Users/24762/Desktop/bop/obj_000000_nocs.ply')
mesh = pyrender.Mesh.from_trimesh(mesh)

# R,T和相机参数
gt_json_path = 'C:/Users/24762/Desktop/bop/scene_gt.json'
camera_json_path = 'C:/Users/24762/Desktop/bop/scene_camera.json'
with open(camera_json_path, 'r') as f:
    scene_camera = json.load(f)
with open(gt_json_path, 'r') as f:
    scene_gt = json.load(f)

# 获得所有的color图片
rgb_images = glob.glob('C:/Users/24762/Desktop/bop/rgb/*.jpg')

for rgb_image in rgb_images:
    image_id = osp.splitext(osp.basename(rgb_image))[0]
    image_id_int = int(image_id)


    cam_K = np.array(scene_camera[str(image_id_int)]['cam_K']).reshape(3, 3)
    cam_pose = np.eye(4)
    cam_pose[1, 1] = -1
    cam_pose[2, 2] = -1


    obj_pose = np.eye(4)
    obj_pose[:3, :3] = np.array(scene_gt[str(image_id_int)][0]['cam_R_m2c']).reshape(3, 3)
    obj_pose[:3, 3] = scene_gt[str(image_id_int)][0]['cam_t_m2c']


    light_itensity = 0.5
    ambient_light = np.array([0.02, 0.02, 0.02, 1.0])  # np.array([1.0, 1.0, 1.0, 1.0])
    if light_itensity != 0.6:
        ambient_light = np.array([1.0, 1.0, 1.0, 1.0])
    scene = pyrender.Scene(
        bg_color=np.array([255.0, 255.0, 255.0, 255.0]), ambient_light=ambient_light
    )
    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=light_itensity,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    scene.add(light, pose=cam_pose)

    camera = pyrender.IntrinsicsCamera(
        fx=cam_K[0, 0], fy=cam_K[1, 1], cx=cam_K[0, 2], cy=cam_K[1, 2], znear=0.05, zfar=100000
    )
    scene.add(camera, pose=cam_pose)

    render_engine = pyrender.OffscreenRenderer(1920, 1080)  # 不同图片参数需要修改
    cad_node = scene.add(mesh, pose=np.eye(4), name="cad")

    # 渲染 NOCS map
    scene.set_pose(cad_node, obj_pose)
    rgb, depth = render_engine.render(scene, pyrender.constants.RenderFlags.RGBA)
    rgb = Image.fromarray(np.uint8(rgb))
    filename =image_id+'_coord'
    rgb.save(f'C:/Users/24762/Desktop/bop/nocs_map/{filename}.png')  #保存为imgae_id + _coord.png 的形式



"""Step 2: mask翻转 """
mask_input_dir = 'C:/Users/24762/Desktop/bop/mask'
mask_output_dir = 'C:/Users/24762/Desktop/bop/mask_inverted'
if not os.path.exists(mask_output_dir):
    os.makedirs(mask_output_dir)

for filename in os.listdir(mask_input_dir):
    if filename.endswith('.png'):

        img_path = os.path.join(mask_input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Change all black (0) values to 255 and all white (255) values to 1
        img[img == 255] = 1
        img[img == 0] = 255

        output_img_path = os.path.join(mask_output_dir, filename)
        cv2.imwrite(output_img_path, img)
# 重新命名mask图片
mask_images =glob.glob('C:/Users/24762/Desktop/bop/mask_inverted/*.png')
for mask_image in mask_images:
    # Get the base name of the image (e.g., '000001_000000.png')
    base_name = os.path.basename(mask_image)
    # Split the base name into the image ID and the rest (e.g., ['000001', '000000.png'])
    parts = base_name.split('_')
    # Construct the new name (e.g., '000001_mask.png')
    new_name = parts[0] + '_mask.png'
    # Get the full path of the new name
    new_path = os.path.join(os.path.dirname(mask_image), new_name)
    # # Rename the image
    os.rename(mask_image, new_path)



"""Step 3: 生成meta.txt文件"""
meta_dir ='C:/Users/24762/Desktop/bop/meta/'
if not os.path.exists(meta_dir):
    os.makedirs(meta_dir)

rgb_path ='C:/Users/24762/Desktop/bop/rgb/'
i=0
filelist = os.listdir(rgb_path)
for file in filelist:
    im = PIL.Image.open(rgb_path + filelist[i])
    meta_filename = os.path.splitext(file)[0]+'_meta'
    meta_path = meta_dir + meta_filename + '.txt'
    meta_create = open(meta_path, 'w')
    meta_create.write('1 1 box_norm ')
    i = i + 1

"""Step 4: 修改为NOCS 格式"""

# 创建新的NOCS文件夹(将文件保存到nocs文件夹中）
NOCS_path = 'C:/Users/24762/Desktop/bop/NOCS/scene_1/'
if not os.path.exists(NOCS_path):
    os.makedirs(NOCS_path)
# 将depth保存到nocs文件夹中
depth_path = 'C:/Users/24762/Desktop/bop/depth_nerf/'
filelist = os.listdir(depth_path)
for i , file in enumerate(filelist):
    im = PIL.Image.open(depth_path+filelist[i])
    filename = os.path.splitext(file)[0] + '_depth'
    save_path = os.path.join(NOCS_path, f'{filename}.png')
    im.save(save_path)

# 将rgb保存到nocs文件夹中
filelist = os.listdir(rgb_path)
for i, file in enumerate(filelist):
    im = PIL.Image.open(rgb_path+filelist[i])
    filename = os.path.splitext(file)[0] + '_color'
    save_path = os.path.join(NOCS_path, f'{filename}.png')
    im.save(save_path)

# 将mask_inverted,meta,nocs_map中的文件复制到NOCS路径下
source_path ={'C:/Users/24762/Desktop/bop/mask_inverted','C:/Users/24762/Desktop/bop/meta','C:/Users/24762/Desktop/bop/nocs_map'}
for dir_path in source_path:
    # 获取源文件夹中的所有文件
    filelist = os.listdir(dir_path)
    # 遍历每个文件
    for filename in filelist:
        # 创建源文件和目标文件的完整路径
        source_file = os.path.join(dir_path, filename)
        target_file = os.path.join(NOCS_path, filename)
        # 复制文件
        shutil.copy2(source_file, target_file)

# 在NOCS文件夹下，有'_color.png','_coord.png','_mask.png','_depth.png','_meta.txt'

"""Step 5: 生成gt_pkl文件（针对验证集）"""

output_gt_dir = 'C:/Users/24762/Desktop/bop/gt'

if not os.path.exists(output_gt_dir):
    os.makedirs(output_gt_dir)

NOCS_ids = [str(int(f.split('_')[0])) for f in os.listdir(NOCS_path) if f.endswith('_color.png')]
for color_id in NOCS_ids:
    if color_id in scene_gt:
        # 对于每个ID，可能有多个RT信息，所以需要遍历它们
        gt_RTs =[]
        for rt_info in scene_gt[color_id]:
            # 从JSON数据中获取旋转矩阵和平移向量
            cam_R_m2c = np.array(rt_info['cam_R_m2c']).reshape(3, 3)
            cam_t_m2c = np.array(rt_info['cam_t_m2c']).reshape(3, 1)/100

            # 创建一个4x4的RT矩阵
            RT = np.eye(4)
            RT[:3, :3] = cam_R_m2c
            RT[:3, 3] = cam_t_m2c.flatten()
            gt_RTs.append(RT)

        gt_RTs = np.array(gt_RTs)

        save_dict = {
            'gt_RTs': gt_RTs,
            'image_path': os.path.join(NOCS_path, f"{color_id}_color.png")
        }
        # 保存RT矩阵到pkl文件
        pkl_file_path = os.path.join(output_gt_dir, f'results_val_scene_1_{int(color_id):04d}.pkl')
        with open(pkl_file_path, 'wb') as pkl_file:
            pickle.dump(save_dict, pkl_file)
    else:
        print(f"No RT information for ID {color_id}")

print("RT matrices have been saved to pkl files.")

"""Step 5: 重新命名NOCS文件"""
suffixes = ['_color.png', '_mask.png', '_depth.png', '_coord.png','_meta.txt']
file_ids = set(os.path.splitext(f)[0].split('_')[0] for f in os.listdir(NOCS_path) if any(f.endswith(suffix) for suffix in suffixes))

# 根据旧的编号对文件进行排序，确保按顺序重命名
sorted_file_ids = sorted(file_ids, key=lambda x: int(x))

# 重命名文件
for idx, old_id in enumerate(sorted_file_ids):
    for suffix in suffixes:
        old_filename = os.path.join(NOCS_path, old_id + suffix)
        # 格式化新文件名为4位数字序列
        new_filename = os.path.join(NOCS_path, f"{idx:04d}" + suffix)
        # 如果旧文件存在，则重命名
        if os.path.exists(old_filename):
            os.rename(old_filename, new_filename)

print(f"Files in {NOCS_path} have been renamed.")

"""Step 6: 获得obj_model中的bbox.txt"""
plyfile = 'C:/Users/24762/Desktop/bop/obj_000000.ply'
obj_models = 'C:/Users/24762/Desktop/bop/NOCS/obj_models'
if not os.path.exists(obj_models):
    os.makedirs(obj_models)

plydata = PlyData.read(plyfile)
# 获取顶点数据
vertices = plydata['vertex']
# 提取x, y, z坐标
x = vertices['x']
y = vertices['y']
z = vertices['z']
# 计算每个维度上的最小和最大值 ，除以100将单位转化为m，再乘以2
x_min, x_max = min(x)/50, max(x)/50
y_min, y_max = min(y)/50, max(y)/50
z_min, z_max = min(z)/50, max(z)/50

with open(os.path.join(obj_models, 'bbox_norm.txt'), 'w') as f:
    f.write(f'{x_max} {y_max} {z_max}\n')
    f.write(f'{x_min} {y_min} {z_min}\n')

print(f"Files done")
