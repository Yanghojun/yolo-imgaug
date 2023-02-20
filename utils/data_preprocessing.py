from glob import glob
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def copy_files_to_dir(file_paths:list, dir_path:str):
    for path in file_paths:
        file_name = os.path.split(path)[-1]
        shutil.copyfile(path, os.path.join(dir_path, f'{file_name}'))
        
def check_files_existence(paths:list):
    for path in paths:
        if not os.path.exists(path):
            print(f'{path} is not exist', end=' ')
            return False
    return True

def data_split(dir_path:str, save_train_path:str, save_valid_path:str):
    """
        1. Process train_test_split data in dir_path
        2. Make directory based on save_train_path, save_valid_path
        3. copy splited data to each directory

    Args:
        dir_path (str): _description_
        save_train_path (str): _description_
        save_valid_path (str): _description_
    """
    img_paths = glob(os.path.join(dir_path, '*.png'))
    # use img datas to read txt files that have same name
    txt_paths = []
    for path in img_paths:
        txt_paths.append(path[:-3] + 'txt')
        
    assert check_files_existence(txt_paths), 'Error!'
    
    train_img_paths, val_img_paths, train_txt_paths, val_txt_paths = train_test_split(img_paths, txt_paths, random_state=5, test_size=0.2)
    os.makedirs(save_train_path, exist_ok=True), os.makedirs(save_valid_path, exist_ok=True)
    copy_files_to_dir(train_img_paths, save_train_path), copy_files_to_dir(train_txt_paths, save_train_path)
    copy_files_to_dir(val_img_paths, save_valid_path), copy_files_to_dir(val_txt_paths, save_valid_path)
    
    
def get_files_by_classId(dir_path:str, class_id:int)->str:
    """Find label files satisfying enterd class_id

    Args:
        dir_path (str): data source directory
        class_id (int): class id you want to find in txt files

    Returns:
        str: _description_
    """
    save_files = []
    txt_paths = glob(os.path.join(dir_path, '*.txt'))
    img_paths = glob(os.path.join(dir_path, '*.png'))
    
    # read lines
    for path in txt_paths:
        if 'classes.txt' in path:
            continue
        
        with open(path, 'r') as f:
            txts = f.readlines()
            
            # check class in txts
            for txt in txts:
                txt_id = int(txt[0])
                if txt_id == class_id:
                    save_files.append(path)
                    save_files.append(path[:-4] + '.png')
                    break
    
    return save_files

if __name__ == '__main__':
    # Get files based on class id
    # txt_paths = get_files_by_classId('.\las_data_annotated', 2)
    
    # for path in txt_paths:
    #     dest = os.path.split(path)[-1]
    #     dest = os.path.join('las_data_only_junction', dest)
    #     shutil.copyfile(path, dest)
    
    # create train, valid directory and Do data split
    data_split('las_data_only_junction', 
               os.path.join('junction_train'), 
               os.path.join('junction_valid'))