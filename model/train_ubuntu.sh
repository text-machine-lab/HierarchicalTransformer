#TODO as soon as training is finished

echo "######################### TRANSFORMER #######################################"

python train.py --data=ubuntu --model=MULTI --batch_size=16 --n_epoch=1 --eval_batch_size=16 \
   --encoder_hidden_size=512 --decoder_hidden_size=512  --embedding_size=512 --context_size=512 \
   --encoder_type=transformer --decoder_type=transformer --tg_enable --msg=full --max_history=140 \
   --full_samples_file=../data/transformer_ubuntu_samples1.txt --max_samples=10000 --save_path=../data/trans_ubuntu_ckpt

: '

echo "######################### UNET ################################"

python train.py --data=ubuntu --model=MULTI --batch_size=16 --n_epoch=1 --eval_batch_size=16 \
   --encoder_hidden_size=512 --decoder_hidden_size=512  --embedding_size=512 --context_size=512 \
   --encoder_type='unet' --decoder_type='transformer' --tg_enable --msg='full' --max_history=150 \
   --full_samples_file=../data/unet_ubuntu_samples1.txt --max_samples=10000

echo "#################### HRED ############################"

python train.py --data=ubuntu --model=HRED --batch_size=16 --n_epoch=1 --eval_batch_size=16 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --max_history=150 --context_size=256 \
   --tg_enable --msg=full --full_samples_file=../data/hred_ubuntu_samples1.txt --max_samples=10000 \
   --save_path=../data/hred_ubuntu_ckpt

echo "#################### GRU #############################"

python train.py --data=ubuntu --model=MULTI --batch_size=16 --n_epoch=1 --eval_batch_size=16 \
   --encoder_hidden_size=512 --decoder_hidden_size=512  --embedding_size=512 --context_size=512 \
   --encoder_type=gru --decoder_type=gru --tg_enable --msg=full --max_history=150 \
   --full_samples_file=../data/gru_ubuntu_samples1.txt --max_samples=10000 --save_path=../data/gru_ubuntu_ckpt

echo "######################### VHCR ################################"

python train.py --data=ubuntu --model=VHCR --batch_size=16 --n_epoch=1 --eval_batch_size=16 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type=unet --decoder_type=transformer --tg_enable --msg=full --max_history=150 \
   --full_samples_file=../data/vhcr_ubuntu_samples1.txt --max_samples=10000

echo "######################### VHRED ################################"

python train.py --data=ubuntu --model=VHRED --batch_size=16 --n_epoch=1 --eval_batch_size=16 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type=unet --decoder_type=transformer --tg_enable --msg=full --max_history=150 \
   --full_samples_file=../data/vhred_ubuntu_samples1.txt --max_samples=10000

'


