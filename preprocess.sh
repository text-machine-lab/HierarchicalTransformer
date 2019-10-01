# python preprocess.py -train_src wmt14_en_fr/baby/baby_train.en -train_tgt wmt14_en_fr/baby/baby_train.fr \
#    -valid_src wmt14_en_fr/baby/baby_val.en -valid_tgt wmt14_en_fr/baby/baby_val.fr -save_data data/babywmt14.pt \
#    -min_word_count=5

python preprocess.py -train_src twitter/train_histories.txt -train_tgt twitter/train_responses.txt \
                     -valid_src twitter/val_histories.txt   -valid_tgt twitter/val_responses.txt \
                     -test_src twitter/test_histories.txt   -test_tgt twitter/test_responses.txt \
                     -save_data data/twitter.pt -min_word_count=5 -share_vocab