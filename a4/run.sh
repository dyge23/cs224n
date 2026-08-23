# bash
export NLTK_ALLOW_PROXIED_URLOPEN=1
uv run run.py train \
--train-src=./zh_en_data/train.zh \
--train-tgt=./zh_en_data/train.en \
--dev-src=./zh_en_data/dev.zh \
--dev-tgt=./zh_en_data/dev.en \
--vocab=./vocab.json \
--cuda \
--resume \
--save-to=./outputs/model.bin

