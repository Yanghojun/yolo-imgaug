import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import xml.etree.ElementTree as ET
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import os
from glob import glob
from tqdm import tqdm
import argparse
    
    
def create_aug_data(data_dir:str, save_dir:str, n_samples=1)->None:
    """Create Augmentated data (based Imgaug github Repo)

    Args:
        n_samples (int): Create n data. If you enter 2, you will get two data of augmentated data for each file
        data_dir (str): data directory that has .png files and .txt files based on yolo label format
        save_dir (str): directory path you want to save augmentated .png & .txt files
    """
    img_paths = glob(os.path.join(data_dir, '*.png'))
    for n in tqdm(range(n_samples), ascii=True):
        for path in tqdm(img_paths, ascii=True):
            img = cv2.imread(path)
            img_height, img_width, channel = img.shape
            
            # read labels
            with open(path[:-4] + '.txt') as f:
                labels = f.readlines()
                bboxes = []
                
                # make new axis. For expanding dimension
                input_img = img[np.newaxis, :, :, :]
                    
                for label in labels:
                    # class_id, x_center, y_center, bbox_width, bbox_height
                    class_id, x_center, y_center, bbox_width, bbox_height = map(float, label.split())
                    
                    abs_x_center, abs_y_center = x_center * img_height, y_center * img_width
                    abs_height = bbox_height * img_height
                    abs_width = bbox_width * img_width
                    
                    # change normalized coordinate to exact coordinate
                    x1 = (abs_x_center - abs_width / 2)
                    y1 = (abs_y_center - abs_height / 2)
                    x2 = (abs_x_center + abs_width / 2)
                    y2 = (abs_y_center + abs_height / 2)
                    
                    bboxes.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=class_id))
                
                # apply augmentation
                # you can apply imgaug function in here to make various data
                # refer here! (https://imgaug.readthedocs.io/en/latest/source/examples_basics.html)
                seq = iaa.Sequential([
                        iaa.Affine(
                        scale={"x": (0.4, 0.8), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-25, 25),
                        shear=(-8, 8)
                        ),
                        # iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))
                ])
                
                # make BoundingBox objects with type(bboxes) == list
                bboxes = BoundingBoxesOnImage(bboxes, shape=img.shape)
                aug_img, aug_bboxes = seq(images=input_img, bounding_boxes=bboxes)
                draw = aug_bboxes.draw_on_image(aug_img[0], size=2, color=[0, 255, 0])  # why aug_img.shape has 4 dim?
                
                # save annotated png files
                aug_img = aug_img.squeeze()
                
                # avoid file overwirting
                uniq = 1
                aug_png_name = os.path.split(path)[-1][:-4]+ '_aug' + f'_{uniq}' + '.png'
                while os.path.exists(os.path.join(save_dir, aug_png_name)):
                    uniq += 1
                    aug_png_name = os.path.split(path)[-1][:-4]+ '_aug' + f'_{uniq}' + '.png'
                cv2.imwrite(os.path.join(save_dir, aug_png_name), aug_img)
                
                # save annotated txt files
                txt_list = []
                for bbox in aug_bboxes:
                    x1, y1, x2, y2, class_id = bbox.x1, bbox.y1, bbox.x2, bbox.y2, bbox.label
                    
                    # change coordinate
                    # (xmin, ymin), (xmax, ymax) -> Normalized (x_c, y_c, width, height)
                    x_c = (x2 - (abs_width / 2)) / img_width
                    y_c = (y2 - (abs_height / 2)) / img_height
                    x_c, y_c = f'{x_c:.3f}', f'{y_c:.3f}'
                    label = f'{int(class_id)} {x_c} {y_c} {bbox_width} {bbox_height}\n'
                    txt_list.append(label)
                
                
                aug_txt_name = os.path.split(path)[-1][:-4]+ '_aug' + f'_{uniq}' + '.txt'
                while os.path.exists(os.path.join(save_dir, aug_txt_name)):
                    uniq += 1
                    aug_txt_name = os.path.split(path)[-1][:-4]+ '_aug' + f'_{uniq}' + '.txt'
                
                with open(os.path.join(save_dir, aug_txt_name), 'w') as ft:
                    ft.writelines(txt_list)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add Augmentation option')
    parser.add_argument('--n', help='Create n data', type=int)
    parser.add_argument('--data_dir', help='Directory path that has png, txt files based on yolo format')
    parser.add_argument('--save_dir', help='Save directory path that you want to save augmented data')
    args = parser.parse_args()
    
    create_aug_data(args.data_dir, args.save_dir, args.n)