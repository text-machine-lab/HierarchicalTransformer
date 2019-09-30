

echo "########################### UNET TRANSFORMER #########################"

python train.py --data=personachat --model=MULTI --batch_size=4 --n_epoch=1000 --eval_batch_size=4 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='unet' --decoder_type='transformer' --tg_enable --msg='full' --max_history=150 \
   --full_samples_file=../data/unet_persona_samples1.txt --max_samples=10000 --num_layers=6

echo "############################### TRANSFORMER ##############################"

python train.py --data=personachat --model=MULTI --batch_size=4 --n_epoch=1000 --eval_batch_size=4 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='transformer' --decoder_type='transformer' --tg_enable --msg='full' --max_history=150 \
   --full_samples_file=../data/transformer_persona_samples1.txt --max_samples=10000 --num_layers=6

: '

echo "#################### VHRED ############################"

python train.py --data=personachat --model=VHRED --batch_size=4 --n_epoch=1000 --eval_batch_size=4 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='unet' --decoder_type='transformer' --tg_enable --msg='full' --max_history=150 \
   --full_samples_file=../data/vhred_persona_samples1.txt --max_samples=10000 --num_layers=1

echo "#################### VHCR ############################"

python train.py --data=personachat --model=VHCR --batch_size=4 --n_epoch=1000 --eval_batch_size=4 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='unet' --decoder_type='transformer' --tg_enable --msg='full' --max_history=150 \
   --full_samples_file=../data/vhcr_persona_samples1.txt --max_samples=10000 --num_layers=1

echo "#################### HRED ############################"

python train.py --data=personachat --model=HRED --batch_size=4 --n_epoch=1000 --eval_batch_size=4 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='unet' --decoder_type='transformer' --tg_enable --msg='full' --max_history=150 \
   --full_samples_file=../data/hred_persona_samples1.txt --max_samples=10000 --num_layers=1

echo "#################### GRU #############################"

python train.py --data=personachat --model=MULTI --batch_size=4 --n_epoch=1000 --eval_batch_size=4 \
   --encoder_hidden_size=256 --decoder_hidden_size=256  --embedding_size=256 --context_size=256 \
   --encoder_type='gru' --decoder_type='gru' --tg_enable --msg='full' --max_history=150 \
   --full_samples_file=../data/gru_persona_samples1.txt --max_samples=10000 --num_layers=1

'