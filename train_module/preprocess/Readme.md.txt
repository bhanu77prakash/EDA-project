This part is taken from https://github.com/facebookresearch/clevr-iep repositiory in github giving its due credit.




===================================================================
Extract Image Features

Extract ResNet-101 features for the CLEVR train, val, and test images with the following commands:

python train_module/preprocess/extract_features.py \
  --input_image_dir data/CLEVR_v1.0/images/train \
  --output_h5_file data/train_features.h5

python train_module/preprocess/extract_features.py \
  --input_image_dir data/CLEVR_v1.0/images/val \
  --output_h5_file data/val_features.h5

python train_module/preprocess/extract_features.py \
  --input_image_dir data/CLEVR_v1.0/images/test \
  --output_h5_file data/test_features.h5
==================================================================
Preprocess Questions

Preprocess the questions and programs for the CLEVR train, val, and test sets with the following commands:

python train_module/preprocess/preprocess_questions.py \
  --input_questions_json data/CLEVR_v1.0/questions/CLEVR_train_questions.json \
  --output_h5_file data/train_questions.h5 \
  --output_vocab_json data/vocab.json

python train_module/preprocess/preprocess_questions.py \
  --input_questions_json data/CLEVR_v1.0/questions/CLEVR_val_questions.json \
  --output_h5_file data/val_questions.h5 \
  --input_vocab_json data/vocab.json
  
python train_module/preprocess/preprocess_questions.py \
  --input_questions_json data/CLEVR_v1.0/questions/CLEVR_test_questions.json \
  --output_h5_file data/test_questions.h5 \
  --input_vocab_json data/vocab.json
===================================================================