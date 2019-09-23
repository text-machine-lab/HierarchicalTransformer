python train.py --data=ubuntu --model=MULTI --batch_size=16 --n_epoch=1 --eval_batch_size=16 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='unet' --decoder_type='transformer' --tg_enable --msg='full' --max_history=150 \
   --full_samples_file=../data/unet_ubuntu_samples1.txt --max_samples=10000