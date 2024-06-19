# UIT-2Q2T at ImageCLEFmedical: Multimodal Medical Image Captioning using Bootstrapping Language-Image Pre-training

**Notebook for the UIT-2Q2T Team at CLEF 2024**

**Thien V. Phan, 21522628@gm.uit.edu.vn**  
**Trinh K. Nguyen, 21522717@gm.uit.edu.vn**  
**Quang A.D.D. Hoang, 21522509@gm.uit.edu.vn**  
**Quan T. Phan, 21522502@gm.uit.edu.vn**  
**Thien B. Nguyen-Tat, thienntb@uit.edu.vn**

## Introduction

In the ImageCLEFmedical 2024 Caption competition, our team utilized the MedBLIP model and achieved significant results. Below are the methods we used for training, predicting, and evaluating our model.

## Training

To train the model, use the following command:

```
python main.py train
    --root_path={path to folder, default: ./}
    --batch_size={number of batches, default: 4}
    --num_epochs={number of epochs, default: 16}
    --lr={learning rate, default: 1e-5}
    --log_wandb={true or false, whether to use wandb for logging}
    --load_weights={true or false, whether to load pre-trained weights}
    --path_weights={path to weights folder, default: ./}
```

Note: The weight file name will be saved automatically and can be adjusted in the code. It will be saved in path_weights, which serves as both the load and save path for continuous training. The file will be named medblip_large.pth.

## Prediction

To generate predictions, use the following command:

```
python main.py predict
    --root_path={path to folder, default: ./} \
    --path_weights={path to the weights file, default: ./medblip_large.pth}
```

Note: The path_weights parameter in predict should be the path to the weight file.
The prediction output will consist of two files: run.csv and valid.csv, which correspond to the predicted captions for the test and validation sets, respectively.

## Evaluation

To evaluate the model, use the following command:

```
python evaluate.py eval
    --root={path to the dataset for evaluation}
    --score={type of score: rouge, bleu, meteor, bertscore}

```
