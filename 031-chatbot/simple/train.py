import tensorflow as tf
from .data import Dataset
from .model import Seq2Seq


def main():
  ds = Dataset()
  ds.load()
  epochs = 100

  for epoch in range(1, epochs + 1):
    for batch_index, (padded_question_in_batch,
                      padded_answers_in_batch) in enumerate(ds.get_batches(25)):
      print('epoch:', epoch,
            'checking batch index', batch_index,
            ' padded quest', padded_question_in_batch[0],
            'padded ans', padded_answers_in_batch[0])


if __name__ == "__main__":
    main()
