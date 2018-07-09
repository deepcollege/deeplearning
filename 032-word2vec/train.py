from model import Word2vec


def main():
  hparams = {
    'epochs': 10000,
    'batch_size': 20,
    'embedding_size': 2,
    'num_sampled': 15,
    'vocab_size': 0,
    'gpu_dynamic_memory_growth': False
  }
  w2v = Word2vec()
  print('checking w2v', w2v)

if __name__ == "__main__":
    main()
