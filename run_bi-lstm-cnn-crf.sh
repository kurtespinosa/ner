THEANO_FLAGS='floatX=float32' python bi_lstm_cnn_crf.py --fine_tune --embedding glove --oov embedding --update momentum \
 --batch_size 10 --num_units 200 --num_filters 30 --learning_rate 0.01 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout \
 --train "data/train" --dev "data/dev" --test "data/test" \
 --embedding_dict "emb/twitter/glove.twitter.27B.100d.txt" --patience 5
