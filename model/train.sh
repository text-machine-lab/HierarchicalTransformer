python train.py --data=ubuntu --model=MULTI --batch_size=32 --n_epoch=1 --eval_batch_size=32 \
   --encoder_hidden_size=128 --decoder_hidden_size=128  --embedding_size=128 --context_size=128 \
   --encoder_type='transformer' --decoder_type='transformer'  --msg='dev' --max_history=300