THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1 python bi_lstm_cnn_crf6.py --fine_tune --embedding glove --oov embedding --update momentum \
 --batch_size 10 --num_units 200 --num_filters 30 --learning_rate 0.01 --decay_rate 0.05 --grad_clipping 5 --regular none --dropout \
 --train "data/train" --dev "data/dev" --test "data/test" \
 --embedding_dict "emb/news/glove.840B.300d.txt" --patience 5 --output_prediction
