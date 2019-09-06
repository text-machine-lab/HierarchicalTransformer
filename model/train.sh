python train.py --data=cornell --model=HRED --batch_size=32 --n_epoch=1 --eval_batch_size=32 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='unet' --decoder_type='transformer'  --msg='dev' --max_history=150 \
   --full_samples_file='../data/full_samples.txt' --max_samples=100 --max_examples=1000 \
   --save_path=../data/unet_ckpt