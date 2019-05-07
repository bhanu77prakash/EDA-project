First run the preprocess part

then run the following command for training.

python scripts/train_model.py \
  --model_type PG \
  --num_train_samples 18000 \
  --num_iterations 20000 \
  --checkpoint_every 1000 \
  --checkpoint_path data/program_generators.pt

python scripts/train_model.py \
  --model_type EE \
  --program_generator_start_from data/program_generator.py \
  --num_iterations 100000 \
  --checkpoint_path data/execution_engine.pt