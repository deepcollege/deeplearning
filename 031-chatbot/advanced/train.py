import os
import argparse
import tensorflow as tf
from .nmt import nmt


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # Output location
    parser.add_argument("--output", type=str, default="/output", help="""\
          example drive/chatbot/output | /output
          Use drive if you are running on Colab
          Use /output if you are running on Floydhub\
          """)

    # Input location
    parser.add_argument("--input", type=str, default="/inputs", help="""\
          example drive/chatbot/input | /inputs
          Use drive if you are running on Colab
          Use /inputs if you are running on Floydhub\
          """)


def main():
  nmt_parser = argparse.ArgumentParser()
  add_arguments(nmt_parser)
  FLAGS, unparsed = nmt_parser.parse_known_args()

  hparams = {
    'attention': 'scaled_luong',
    'num_train_steps': 10000000,
    'num_layers': 2,
    #    'num_encoder_layers': 2,
    #    'num_decoder_layers': 2,
    'num_units': 512,
    #    'batch_size': 128,
    #    'override_loaded_hparams': True,
    #    'decay_scheme': 'luong234'
    #    'residual': True,
    'optimizer': 'adam',
    'encoder_type': 'bi',
    'learning_rate': 0.001,
    'beam_width': 20,
    'length_penalty_weight': 1.0,
    'num_translations_per_input': 20,
    #    'num_keep_ckpts': 5,

    ## You don't normally need to change anything below (internal settings)
    'src': 'from',
    'tgt': 'to',
    # 'vocab_prefix': os.path.join(train_dir, "vocab"),
    'train_prefix': os.path.join(FLAGS.input, "train"),
    'dev_prefix': os.path.join(FLAGS.input, "tst2012"),
    'test_prefix': os.path.join(FLAGS.input, "tst2013"),
    'out_dir': FLAGS.output,
    # 'share_vocab': preprocessing['joined_vocab'],
  }

  nmt.FLAGS, unparsed = nmt_parser.parse_known_args(['--' + k + '=' + str(v) for k, v in hparams.items()])
  print(os.getcwd() + '/advanced/nmt/nmt.py')
  print(unparsed)
  tf.app.run(main=nmt.main, argv=[os.getcwd() + '\advanced\nmt\nmt.py'] + unparsed)

if __name__ == "__main__":
    main()
