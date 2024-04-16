## A Simple Script for Converting BOP datasets to NOCS dataset format
___
### BOP dataset structure
rgb 彩色图像

mask 物体轮廓的mask

mask_visib 物体轮廓可见部分的mask

depth_nerf 深度图

scene_camera.json 每张图片对应的相机内参

scene_gt.json 每张图片中对应物体的真实的位姿
___
### NOCS dataset structure
{image_id}_color.png 彩色图像

{image_id}_depth.png 深度图像

{image_id}_mask.png 物体轮廓的mask
mask的像素值与物体类别有关

{image_id}_coord.png 物体的NOCS map

{image_id}_meta.txt 物体的序号 类别 3D模型


