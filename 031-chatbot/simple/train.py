import tensorflow as tf
from .data import Dataset
from .model import Seq2Seq


def main():
  # Dataset, default should be using Cornell
  ds = Dataset()
  ds.load()

  # Hyperparams
  epochs = 100
  model_hparams = dict({
    'batch_size': 64,
    'sequence_length': 25,
    'encoding_embedding_size': 512,
    'decoding_embedding_size': 512,
    'rnn_size': 512,
    'num_layers': 3,
    'learning_rate': 0.01,
    'gpu_dynamic_memory_growth': False,
    'keep_probability': 0.5,

    'num_questions_word2count': ds.sub.num_questions_word2count,
    'num_answers_word2count': ds.sub.num_answers_word2count,
    'get_word2int': ds.sub.get_word2int,
  })

  # Compiling model
  model = Seq2Seq(
    model_hparams=model_hparams
  )
  model.compile()

  for epoch in range(1, epochs + 1):
    for batch_index, (padded_question_in_batch,
                      padded_answers_in_batch) in enumerate(ds.get_batches(25)):
      print('epoch:', epoch,
            'checking batch index', batch_index,
            ' padded quest', padded_question_in_batch[0],
            'padded ans', padded_answers_in_batch[0])


if __name__ == "__main__":
    main()
