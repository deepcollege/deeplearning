data_advanced:
	python -m advanced.data

train_advanced_floyd:
	python -m advanced.data --input="/inputs" --output="/output"
	python -m advanced.nmt.nmt \
		--src=vi --tgt=en \
		--vocab_prefix=/inputs/vocab  \
		--train_prefix=/inputs/train \
		--dev_prefix=/inputs/dev  \
		--test_prefix=/inputs/test \
		--out_dir=/output/nmt_model \
		--num_train_steps=12000 \
		--steps_per_stats=100 \
		--num_layers=2 \
		--num_units=128 \
		--dropout=0.2 \
		--metrics=bleu