import os
import PIL
import numpy as np
import matplotlib.pyplot as plt

def clip_image(image, classes, image_name):
	w, h = image.size

	if h>w:
		new = w
	else:
		new = h

	left = (w-new)/2
	top = (h-new)/2
	right = (w+new)/2
	bottom = (h+new)/2

	image = image.crop((left, top, right, bottom))
	image = image.resize((1024, 1024))

	image.save(f"/home/nas/Research_Group/Personal/Dino/dataset_4_1/test/comp_test/{image_name}")

def make_path_list(image_path):
	lst = os.listdir(image_path)
	os.makedirs(f"/home/nas/Research_Group/Personal/Dino/dataset_4_1/test/comp_test", exist_ok=True)
	go_through_data(lst, None)
	

def go_through_data(lst, classes):
	root = "/home/nas/Research_Group/Personal/Dino/dataset_4_1/test/test_0/"
	i = 0
	for img_path in lst:
		i += 1
		img_data = PIL.Image.open(root + img_path)
		clip_image(img_data, classes, img_path)

def main():

	# os.mkdir("/home/ubuntu/crop_identification/crop_model_1/dataset_1/process_train")
	image_path = "/home/nas/Research_Group/Personal/Dino/dataset_4_1/test/test_0"
	make_path_list(image_path)


if __name__ == "__main__":
	main()