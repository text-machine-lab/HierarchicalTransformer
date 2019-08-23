echo "#################### GRU #############################"

python train.py --data=cornell --model=MULTI --batch_size=32 --n_epoch=40 --eval_batch_size=32 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='gru' --decoder_type='gru' --tg_enable --msg='full' | tee ../data/logs/gru_train.txt

echo "################### TRANSFORMER ENCODER / GRU DECODER ###############"

python train.py --data=cornell --model=MULTI --batch_size=32 --n_epoch=40 --eval_batch_size=32 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='transformer' --decoder_type='gru' --tg_enable --msg='full' | tee ../data/logs/tenc-grudec_train.txt

echo "####################### UNET ENCODER / GRU DECODER ######################"

python train.py --data=cornell --model=MULTI --batch_size=32 --n_epoch=40 --eval_batch_size=32 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='unet' --decoder_type='gru' --tg_enable --msg='full' | tee ../data/logs/unetenc-grudec_train.txt

echo "####################### TRANSFORMER #######################################"

python train.py --data=cornell --model=MULTI --batch_size=32 --n_epoch=40 --eval_batch_size=32 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='transformer' --decoder_type='transformer' --tg_enable --msg='full' | tee ../data/logs/unetenc-grudec_train.txt

echo "######################### UNET TRANSFORMER ################################"

python train.py --data=cornell --model=MULTI --batch_size=32 --n_epoch=40 --eval_batch_size=32 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='unet' --decoder_type='transformer' --tg_enable --msg='full' | tee ../data/logs/unetenc-grudec_train.txt