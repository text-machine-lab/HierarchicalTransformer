python train.py --data=ubuntu --model=MULTI --batch_size=16 --n_epoch=1000 --eval_batch_size=16 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='transformer' --decoder_type='transformer'  --msg='dev' --max_history=150 --max_examples=1000
   --clip=100.0