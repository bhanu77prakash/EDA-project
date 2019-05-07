# EDA-project

## Requirements

1. python 3
2. PyTorch 

<i>Please note the versions to be installed</i>
		torch==0.1.11._5                                                                                                   torchvision==0.1.9

## For training 
1. Download the data into train\_module/data 
2. Fowllow the train_module/preprocess/Readme.md
3. Follow the train\_module/Readme.md to run train_model.py file.

## Pretrained models
The models are present in a drive link mentioned below 
Download them and paste them in test_module/models

## For testing 

### Using pretrained models 
1. Download the models from the drive link given above. 
2. Install the requirements by running pip install -r requirements.txt
3. Inside the test_module run the line python3 FE.py for frontend based model
4. Run the following command for graphical based model 
		python run_model.py --image img/CLEVR_val_000013.png --question "Does the small sphere have the same color as the cube left of the gray cube?"
 

### Training from scratch
1. Follow the training section to train the model
2. Then copy the trained models instead of downloading in the above testing step.
3. Follow the rest of the steps mentioned in the testing section

## References 
1. Stackoverflow 
2. Clevr-iep[https://github.com/facebookresearch/clevr-iep]
3. Clevr-iep[https://arxiv.org/abs/1705.03633]
