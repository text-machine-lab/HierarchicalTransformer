python preprocess.py -train_src wmt14_en_fr/baby/baby_train.en -train_tgt wmt14_en_fr/baby/baby_train.fr \
    -valid_src wmt14_en_fr/baby/baby_val.en -valid_tgt wmt14_en_fr/baby/baby_val.fr -save_data data/babywmt14.pt \
    -min_word_count=5