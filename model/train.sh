python train.py --data=cornell --model=TRANSFORMER --batch_size=32 --n_epoch=1000 --eval_batch_size=32 \
   --encoder_hidden_size=512 --decoder_hidden_size=512  --context_size=256 --embedding_size=512 --clip=2000.0 \
   --learning_rate=1.0 --max_convo_len=100 --max_unroll=180 --max_examples=10000

#python train.py --data=cornell --model=HRED --batch_size=32 --n_epoch=30 --eval_batch_size=32 --tg_enable \
#   --encoder_hidden_size=512 --decoder_hidden_size=512  --embedding_size=512 --context_size=512