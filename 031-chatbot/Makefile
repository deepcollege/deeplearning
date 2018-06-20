data_advanced:
	python -m advanced.data --input="/inputs" --output="/output"

train_advanced_floyd:
	python -m advanced.data --input="/inputs" --output="/output"
	python -m advanced.nmt.nmt \
		--src=from \
		--tgt=to \
		--train_prefix=/inputs/train \
		--dev_prefix=/inputs/dev  \
		--test_prefix=/inputs/test \
		--vocab_prefix=/inputs/vocab \
		--out_dir=/output/nmt_model \
		--attention=scaled_luong
		--num_train_steps=10000000 \
		--num_layers=2 \
		--num_units=512 \
		--optimizer=adam \
		--encoder_type=bi \
		--learning_rate=0.001 \
		--beam_width=20 \
		--length_penalty_weight=1.0 \
		--num_translations_per_input=20\
