# Import important libraries
import pandas as pd
import transformers
from transformers import BlipProcessor, BlipForImageTextRetrieval,BlipForConditionalGeneration, AutoProcessor,Blip2ForConditionalGeneration

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
import os

from tqdm import tqdm

import cv2
import wandb
import os
import matplotlib.pyplot as plt
import copy
from dataset import ImgCaptionDataset
import argparse
def save_df_to_csv(df, file_path):
    df.to_csv(file_path, index=False)

def load_df (dir_caption ) :
    """
    Function load dataframe
    """
    return pd.read_csv(dir_caption, delimiter=",")
def train(root_path ,batch_size=4,num_epochs=2,lr = 1e-5,log_wandb= True,load_weights  = False,  path_weights="./",preprocessing = False):

    """
    Function training
    root_path: root forder of dataset
    batch_size: batch size
    num_epochs: number of epochs
    lr: learning rate
    log_wandb: log to wandb
    load_weights: load weights
    path_weights: path for weights
    preprocessing: preprocessing
    
    """
    #Define paths
    train_dir = os.path.join(root_path, "train")


    #root captions
    train_captions = os.path.join(root_path, "train_captions.csv")
    #root concepts
    train_concepts = os.path.join(root_path, "train_concepts.csv")

    #load dataset csv

    df_train = load_df( dir_caption = train_captions)
    

    #load weights
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype= torch.float32)

    #parama
    image_size = (224, 224)
    max_length = 200
    valid_num = 2000
    #load weights if load_weights=True
    if load_weights :
        model.load_state_dict(torch.load(path_weights+"medblip_large.pth"))
        model.eval()
    #set up cuda and optiminzer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    
    if log_wandb:
        os.system('wandb login --relogin d473c273fec9fb826260a9468ff9ea10afee7ff6')
        os.environ['WANDB_PROJECT'] = 'ASR_with_NST'
        wandb.init(project='MedBLIP2', name=f"MedBlip-large-{num_epochs}epochs-{lr}-lr-2nd-run")

    # Create dataset for training
    train_dataset = ImgCaptionDataset(df = df_train,
                                    path = train_dir, 
                                    processor = processor,
                                    image_size = image_size,
                                    max_length = max_length,
                                    preprocessing = preprocessing
                                    )
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)


  
    # # Start training 
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        for batch in tqdm(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            attention_marks = batch.pop("attention_mask").to(device)
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids,
                            attention_mask = attention_marks)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if log_wandb:
                wandb.log({"train/epoch": epoch})
                wandb.log({"train/loss": loss.item()})
                wandb.log({"train/lr": optimizer.param_groups[-1]['lr']})
            
        torch.save(model.state_dict(), path_weights+"medblip_large.pth") 
        print("Loss:", loss.item())
def predict(root_path,path_weights="./"):
    """
    root_path: root forder of dataset
    path_weights: path for weights file
    """
    ## Load weights pretrained
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
   
    model.load_state_dict(torch.load(path_weights))
    model.eval()
    model.to("cuda")


    df_valid = os.path.join(root_path, "valid_captions.csv") 
    df_valid = pd.read_csv(df_valid)
    dir_valid = os.path.join(root_path,"valid")

    dir_test = os.path.join(root_path,"test")
    dir_caption = os.path.join(root_path, "valid_captions.csv")

    test_ID = os.listdir(dir_test)
    for i in range(len(test_ID)):
        test_ID[i] = test_ID[i].replace(".jpg","")
    def get_inferences(IDs, model, paths, max_new_tokens=200):
        """
        Function to get inferences
        """
        data = []
        for ID in tqdm(IDs):
            path = os.path.join(paths, ID + ".jpg")
            image = cv2.imread(path)
            image = cv2.resize(image, (224, 224))
            inputs = processor(image, return_tensors="pt").to("cuda")
            generated_ids = model.generate(**inputs,
                                        max_new_tokens=max_new_tokens,
                                            no_repeat_ngram_size=2,
                                        num_beams = 5)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            data.append([ID, generated_text])

        df = pd.DataFrame(data, columns=['ID', 'Caption'])
        return df
        ## Get inferences test
    test_results = get_inferences(test_ID, model, dir_test)
    save_df_to_csv(test_results, "run.csv")

        ## Get inferences train
    valid_ID = df_valid["ID"]
    valid_results = get_inferences(valid_ID, model, dir_valid)
    save_df_to_csv(valid_results, "valid.csv")
    len(valid_results)

def main():
    # Initializes a parser for command-line arguments.
    parser = argparse.ArgumentParser()

    # Creates subparsers for different commands.
    subparsers = parser.add_subparsers(dest='command')

    # Adds a subparser for the 'train' command.
    parser_train = subparsers.add_parser('train')

    # Adds an argument for the directory containing public data.
    parser_train.add_argument('--root_path', type=str,default='./')

    parser_train.add_argument('--batch_size', type=int,default=4)
    parser_train.add_argument('--num_epochs', type=int,default=16)
    parser_train.add_argument('--lr', type=float,default=1e-5)

    parser_train.add_argument('--log_wandb', type=bool,default=True)
    parser_train.add_argument('--load_weights', type=bool,default=False)
    
    parser_train.add_argument('--path_weights', type=str,default='./')
    # Adds a subparser for the 'predict' command.
    parser_predict = subparsers.add_parser('predict')

    # Adds an argument for the directory containing public data.
    parser_predict.add_argument('--root_path', type=str,default='./')

    parser_predict.add_argument('--path_weights', type=str,default='./')

    # Parses the command-line arguments.
    args = parser.parse_args()


    if args.command == 'train':
        # Checks if the 'train' command was given.
        # Calls the function to train the model.
        train(args.root_path, args.batch_size, args.num_epochs,args.lr, args.log_wandb,args.load_weights,args.path_weights)
        pass
    elif args.command == 'predict':
        # Checks if the 'predict' command was given.
        # Calls the function to predict the model.
        predict(args.root_path, args.path_weights)
        pass
    
# Checks if the script is being run as the main program.
if __name__ == "__main__":
    # Calls the main function if the script is executed directly.
    main()
