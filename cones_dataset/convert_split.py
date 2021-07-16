import os
from shutil import copyfile, copytree, move
import glob
import supervisely_to_Boxes
import cv2
import random
from math import ceil

# directory_list = list()
# for root, dirs, files in os.walk('good', topdown=False):
#     for name in dirs:
#         directory_list.append(os.path.join(root, name))

# print(directory_list)

# lest the subfolders in the condition folder


def toSupervisely(project_path, visualize_one=False):
    for image_name in os.listdir(os.path.join(project_path, "images")):
        supervisely_to_Boxes.superviselyToDarknet(
            cv2.imread(os.path.join(project_path, "images", image_name)).shape,
            os.path.join(project_path, "ann", image_name + ".json"),
            os.path.join(project_path, "labels", image_name.split(".")[0]+".txt"),
        )
    # copytree(os.path.join(project_path, "labels"), os.path.join(project_path, "augmented", "labels"))


# root='good'
# new_root = 'allgood'
# dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]

# img_ind = 1
# ann_ind = 1

# for item in dirlist:
#     old_img_dir = root + '\\' + item + '\\img' + '\\*'
#     old_annot_dir = root + '\\' + item + '\\ann' + '\\*'

#     for file in glob.glob(old_img_dir):
#         # print(file)
#         dst = new_root + '\\images' + '\\' + str(img_ind) + '.png'
#         copyfile(file, dst)
#         img_ind = img_ind + 1

#     for file in glob.glob(old_annot_dir):
#         dst = new_root + '\\ann' + '\\' + str(ann_ind) + '.png' + '.json' 
#         copyfile(file, dst)
#         ann_ind = ann_ind + 1

# to convert the Supervisly annotations to YOLO ones.
toSupervisely('/mnt/mars-beta/izzeddin/testdata')


# # To split the dataset into 70% training and 30% validation
# img_src_dir = 'good/images'
# ann_src_dir = 'good/labels'

# img_lst = [item for item in os.listdir(img_src_dir)]
# ann_list = [item for item in os.listdir(ann_src_dir)]

# validation_ratio = 0.3
# val_size = ceil(validation_ratio * len(img_lst))
# val_ind = random.sample(range(len(img_lst)), k=val_size)

# print(len(img_lst), val_size)

# for ind in val_ind:
#     val_img = img_lst[ind]
#     val_ann = ann_list[ind]

#     img_src = img_src_dir + '/' + val_img
#     img_dst = 'validation/images' + '/' + val_img
#     move(img_src, img_dst)

#     ann_src = ann_src_dir + '/' + val_ann
#     ann_dst = 'validation/labels' + '/' + val_ann
#     move(ann_src, ann_dst)
    
# img_lst = [item for item in os.listdir(img_src_dir)]
# print(len(img_lst))
