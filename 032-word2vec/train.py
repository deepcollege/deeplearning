from model import Word2vec
from data import GOTData


def main():
    got_data = GOTData()
    got_data.load()
    hparams = {
        'epochs': 10000,
        'batch_size': 20,
        'embedding_size': 2,
        'num_sampled': 15,
        'vocab_size': got_data.vocab_size,
        'gpu_dynamic_memory_growth': False,
        'checkpoint': '/output/w2v'
    }
    w2v = Word2vec(**hparams)
    print('Checking W2V initiation params', w2v.toJSON())
    w2v.compile()

    # Potential TODOs
    # 1. save weights according to epoch
    # 2. tfboard
    for epoch in range(hparams['epochs']):
        batch_inputs, batch_labels = got_data.get_batch(
          size=hparams.get('batch_size'))
        loss_val = w2v.train_batch(batch_inputs, batch_labels)

        if epoch % 1000 == 0:
          print('Loss at ', epoch, loss_val)
          w2v.save_model(hparams.get('checkpoint'))


if __name__ == "__main__":
    main()
