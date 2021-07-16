# this is the code for converting the annotations from supervisly format to YOLO format or COCO format.
# also, it splits the dataset into training and validation.





def toSupervisely(project_path, visualize_one=False):
    for image_name in os.listdir(os.path.join(project_path, "images")):
        supervisely_to_Boxes.superviselyToDarknet(
            cv2.imread(os.path.join(project_path, "images", image_name)).shape,
            os.path.join(project_path, "ann", image_name + ".json"),
            os.path.join(project_path, "labels", image_name.split(".")[0]+".txt"),
        )
    shutil.copytree(os.path.join(project_path, "labels"), os.path.join(project_path, "augmented", "labels"))
