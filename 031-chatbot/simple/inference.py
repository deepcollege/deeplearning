import tensorflow as tf
import argparse
from .data import Dataset
from .model import Seq2Seq
import os



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

seq2seq_parser = argparse.ArgumentParser()
add_arguments(seq2seq_parser)
FLAGS, _ = seq2seq_parser.parse_known_args()

print('Initaiting the training with the following FLAGS')
print(FLAGS)

# Dataset, default should be using Cornell
ds = Dataset(FLAGS)
ds.load()

# Hyperparams
batch_size = 32
epochs = 100
learning_rate = 0.001
learning_rate_decay = 0.9
min_learning_rate = 0.0001
batch_index_check_training_loss = 100
batch_index_check_validation_loss = (
                                    ds.sub.num_questions_word2count // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
model_hparams = dict({
  # Actual hyperparameters
  'batch_size': batch_size,
  'sequence_length': 25,
  'encoding_embedding_size': 1024,
  'decoding_embedding_size': 1024,
  'rnn_size': 1024,
  'num_layers': 3,
  'gpu_dynamic_memory_growth': False,
  'keep_probability': 0.5,
  'learning_rate': learning_rate,

  # static values
  'num_questions_word2count': ds.sub.num_questions_word2count,
  'num_answers_word2count': ds.sub.num_answers_word2count,
  'get_word2int': ds.sub.get_word2int,
})

# Compiling model

model = Seq2Seq(model_hparams=model_hparams, FLAGS=FLAGS)
model.compile()

# Loading the weights and Running the session

cwd = os.getcwd()
print('Loading file from', cwd)
output_checkpoint = cwd + '/simple/output/chatbot_weights.ckpt'
# session = tf.InteractiveSession()
# print('Interactive session enabled')
# session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(model.session, output_checkpoint)


def sample_reply(model, ds, question):
  answer = model.inference(
    question=question,
    questions_words_2_ints=ds.sub.questions_words_2_counts,
    answers_ints_2_words=ds.sub.answers_counts_2_words)
  print(
  'Question: {question}\n' 'Answer: {answer}'.format(question=question,
                                                     answer=answer))


while(True):
    question = raw_input('You: ')
    sample_reply(model, ds, question)
