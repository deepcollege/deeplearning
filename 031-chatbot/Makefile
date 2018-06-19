data_advanced:
	python -m advanced.data

train_advanced_floyd:
	python -m advanced.data
	python -m advanced.nmt.nmt \
		--src=vi --tgt=en \
		--vocab_prefix=/input/vocab  \
		--train_prefix=/input/train \
		--dev_prefix=/input/dev  \
		--test_prefix=/input/tst2013 \
		--out_dir=/tmp/nmt_model \
		--num_train_steps=12000 \
		--steps_per_stats=100 \
		--num_layers=2 \
		--num_units=128 \
		--dropout=0.2 \
		--metrics=bleu