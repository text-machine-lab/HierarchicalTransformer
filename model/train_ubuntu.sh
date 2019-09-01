#TODO as soon as training is finished

echo "######################### UNET TRANSFORMER ################################"

python train.py --data=ubuntu --model=MULTI --batch_size=16 --n_epoch=1 --eval_batch_size=16 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='unet' --decoder_type='transformer' --tg_enable --msg='full' --max_history=150 \
   --full_samples_file=../data/unet_ubuntu_samples1.txt --max_samples=10000

: '

echo "######################### TRANSFORMER #######################################"

python train.py --data=ubuntu --model=MULTI --batch_size=16 --n_epoch=100 --eval_batch_size=16 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='transformer' --decoder_type='transformer' --tg_enable --msg='full' --max_history=150 \
   --full_samples_file=../data/transformer_ubuntu_samples1.txt --max_samples=10000 --save_path=../data/trans_ubuntu_ckpt

echo "#################### HRED ############################"

python train.py --data=ubuntu --model=HRED --batch_size=16 --n_epoch=100 --eval_batch_size=16 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --max_history=150 --context_size=256 \
   --tg_enable --msg='full' --full_samples_file=../data/hred_ubuntu_samples1.txt --max_samples=10000 --save_path=../data/hred_ubuntu_ckpt

echo "#################### GRU #############################"

python train.py --data=ubuntu --model=MULTI --batch_size=16 --n_epoch=100 --eval_batch_size=16 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='gru' --decoder_type='gru' --tg_enable --msg='full' --max_history=150 \
   --full_samples_file=../data/gru_ubuntu_samples1.txt --max_samples=10000 --save_path=../data/gru_ubuntu_ckpt

'


