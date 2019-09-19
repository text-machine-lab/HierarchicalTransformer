# python twitter_train.py -data data/twitter.pt -save_model trained -save_mode best -proj_share_weight -label_smoothing \
#     -epoch=1 -n_warmup_steps=4000 -lr_factor=1.0 -d_model=256 -d_inner_hid=1024 -n_layers=6 -unet -batch_size=64

# python wmt_train.py -data data/wmt14_sample/config.json -save_model saved_models/wmt_sample -save_mode best -proj_share_weight -label_smoothing \
#     -epoch=1 -n_warmup_steps=4000 -lr_factor=1.0 -d_model=256 -d_inner_hid=1024 -n_layers=6 -unet -batch_size=64 -wmt

# python twitter_train.py -data data/wmt14_sample.pt -save_model trained -save_mode best -proj_share_weight -label_smoothing \
#     -epoch=1 -n_warmup_steps=4000 -lr_factor=1.0 -d_model=256 -d_inner_hid=1024 -n_layers=6 -unet -batch_size=64

python wmt_train.py -data data/wmt14/config.json -save_model saved_models/wmt_ -save_mode best -proj_share_weight -label_smoothing \
    -epoch=1 -n_warmup_steps=4000 -lr_factor=1.0 -d_model=256 -d_inner_hid=1024 -n_layers=6 -unet -batch_size=64 -wmt
