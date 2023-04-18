import pickle
import cv2
import lmdb
import os

env = lmdb.open(str("IAM/lmdb"), map_size= 1024 * 1024 * 1024 * 2)

with env.begin(write=True) as txn:
    for dirpath,dirname,filenames in os.walk(r'IAM\img'):
        for files in filenames:
            img_path = dirpath+'\\'+files
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            base_name = os.path.basename(img_path)
            txn.put(base_name.encode("ascii"),pickle.dumps(img))
            
env.close()