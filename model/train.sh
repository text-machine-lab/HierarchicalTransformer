python train.py --data=cornell --model=TRANSFORMER --batch_size=32 --n_epoch=1 --eval_batch_size=32 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --context_size=256 --embedding_size=256 --clip=300.0 \
   --learning_rate=1.0 --max_convo_len=100