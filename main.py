# Import important libraries
import pandas as pd
import transformers
from transformers import BlipProcessor, BlipForImageTextRetrieval,BlipForConditionalGeneration, AutoProcessor,Blip2ForConditionalGeneration

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
import os

from tqdm import tqdm


import wandb
import os
import matplotlib.pyplot as plt
import copy
from dataset import ImgCaptionDataset
import argparse
def load_df (dir_caption ) :
    return pd.read_csv(dir_caption, delimiter=",")
def train(root_path ,batch_size=4,num_epochs=2,lr = 1e-5,log_wandb= True,load_weights  = False,  path_weights="./",preprocessing = False):

    """
    Function training
    root_path: root forder of dataset
    
    """
    #Define paths
    train_dir = os.path.join(root_path, "train")


    #root captions
    train_captions = os.path.join(root_path, "train_captions.csv")
    #root concepts
    train_concepts = os.path.join(root_path, "train_concepts.csv")

    #load dataset csv

    df_train = load_df( dir_caption = train_captions, 
                      dir_concepts = train_concepts,
                  )
    

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
    # Parses the command-line arguments.
    args = parser.parse_args()


    if args.command == 'train':
        # Checks if the 'train' command was given.
        # Calls the function to train the model.
        train(args.root_path, args.batch_size, args.num_epochs,args.lr, args.log_wandb,args.load_weights,args.path_weights)
        pass
    
# Checks if the script is being run as the main program.
if __name__ == "__main__":
    # Calls the main function if the script is executed directly.
    main()
