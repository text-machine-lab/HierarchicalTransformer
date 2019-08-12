python train.py --data=cornell --model=TRANSFORMER --batch_size=32 --n_epoch=1000 --eval_batch_size=32 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --context_size=256 --embedding_size=256 --clip=2000.0 \
   --learning_rate=1.0 --max_convo_len=100 --max_examples=1000 --max_unroll=180