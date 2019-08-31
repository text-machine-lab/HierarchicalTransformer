python train.py --data=ubuntu --model=HRED --batch_size=16 --n_epoch=1 --eval_batch_size=16 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='unet' --decoder_type='transformer'  --msg='dev' --max_history=150 --full_samples_file='../data/unet_full_samples.txt' --max_samples=100