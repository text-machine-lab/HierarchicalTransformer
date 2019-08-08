python train.py --data=cornell --model=TRANSFORMER --batch_size=32 --n_epoch=100 --eval_batch_size=32 \
   --encoder_hidden_size=512 --decoder_hidden_size=512  --context_size=512 --embedding_size=512 --clip=200.0 \
   --learning_rate=1e-3 --max_convo_len=100

echo "###################################################"

python train.py --data=cornell --model=HRED --batch_size=32 --n_epoch=100 --eval_batch_size=32

echo "###################################################"

python train.py --data=cornell --model=TRANSFORMER --batch_size=32 --n_epoch=100 --eval_batch_size=32 \
   --encoder_hidden_size=512 --decoder_hidden_size=512  --context_size=512 --embedding_size=512 --clip=200.0 \
   --learning_rate=1e-3 --unet --max_convo_len=100