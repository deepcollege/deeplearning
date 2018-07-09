from model import Word2vec


def main():
    hparams = {
        'epochs': 10000,
        'batch_size': 20,
        'embedding_size': 2,
        'num_sampled': 15,
        'vocab_size': 10,
        'gpu_dynamic_memory_growth': False
    }
    w2v = Word2vec(**hparams)
    print('Checking W2V initiation params', w2v.toJSON())

    # Potential TODOs
    # 1. save weights according to epoch
    # 2. tfboard
    # 3.
    for epoch in range(hparams['epochs']):
        print(epoch)


if __name__ == "__main__":
    main()
