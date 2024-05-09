
import cv2
import os
import copy
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ImgCaptionDataset(Dataset):
    def __init__(self, 
                 df, 
                 path,
                 processor=None,
                 preprocessing = False,
                 image_size=(224, 224),
                 max_length=100):
        self.df = df 
        self.image_size = image_size 
        self.max_length = max_length
        self.processor = processor
        self.preprocessing = preprocessing
        self.path = path
        self.ids = list(self.df["ID"])      
    def get_caption_by_id(self, iid):
        return list(self.df["Caption"][self.df['ID'] == iid])
    
    def get_image_by_id(self, iid):
        if self.preprocessing == True: 
            image = self.preprocess(iid)
        else: 
            img_path = os.path.join(self.path, str(iid) + ".jpg")
            image = cv2.imread(img_path)
        return image
    
    def get_concepts_by_id(self, iid):
        return list(self.df["Concepts"][self.df["ID"] == iid])[0]
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        iid = self.df["ID"][idx]
        image = self.get_image_by_id(iid) 
        image = cv2.resize(image,self.image_size)
        caption = self.get_caption_by_id(iid)
        concepts = ";".join(self.get_concepts_by_id(iid))
        encoding = self.processor(images=image, 
                                  text= caption, 
                                  padding="max_length",
                                  truncation = "longest_first",
                                  max_length = self.max_length,
                                  return_tensors="pt")
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding