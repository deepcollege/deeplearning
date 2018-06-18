import time
import random
import argparse
from .data import Dataset
from .model import Seq2Seq

# Dummy
sample_questions = ['hello', 'how are you?', 'what do you want?', 'who are you?', 'I am your creator']


def sample_reply(model, ds):
    sample_question = random.choice(sample_questions)
    sample_answer = model.inference(
        question=sample_question,
        questions_words_2_ints=ds.sub.questions_words_2_counts,
        answers_ints_2_words=ds.sub.answers_counts_2_words)
    print('Question: {question}\n' 'Answer: {answer}'.format(question=sample_question, answer=sample_answer))


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
    seq2seq_parser = argparse.ArgumentParser()
    add_arguments(seq2seq_parser)
    FLAGS, _ = seq2seq_parser.parse_known_args()

    print('Initaiting the training with the following FLAGS')
    print(FLAGS)

    # Dataset, default should be using Cornell
    ds = Dataset(FLAGS)
    ds.load()

    # Model savepoint
    checkpoint = '{output_dir}/chatbot_weights.ckpt'.format(output_dir=FLAGS.output)

    # Hyperparams
    batch_size = 128
    epochs = 100
    learning_rate = 2.0
    learning_rate_decay = 0.99
    min_learning_rate = 0.1
    batch_index_check_training_loss = 100
    batch_index_check_validation_loss = (ds.sub.num_questions_word2count // batch_size // 2) - 1
    total_training_loss_error = 0
    list_validation_loss_error = []
    early_stopping_check = 0
    early_stopping_stop = 1000
    model_hparams = dict({
    # Actual hyperparameters
        'batch_size': batch_size,
        'sequence_length': 25,
        'encoding_embedding_size': 256,
        'decoding_embedding_size': 256,
        'rnn_size': 256,
        'num_layers': 2,
        'gpu_dynamic_memory_growth': False,
        'keep_probability': 0.5,
        'learning_rate': learning_rate,

    # static values
        'num_questions_word2count': ds.sub.num_questions_word2count,
        'num_answers_word2count': ds.sub.num_answers_word2count,
        'get_word2int': ds.sub.get_word2int,
    })

    # Compiling model
    model = Seq2Seq(model_hparams=model_hparams)
    model.compile()

    for epoch in range(1, epochs + 1):
        sample_reply(model, ds)
        for batch_index, (padded_question_in_batch, padded_answers_in_batch) in enumerate(ds.get_batches(batch_size)):

            starting_time = time.time()
            batch_training_loss_error = model.train_batch(
                inputs=padded_question_in_batch, targets=padded_answers_in_batch, learning_rate=learning_rate)
            total_training_loss_error += batch_training_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time

            # Smapling to check current progress
            # At every batch_index_check_training_loss (e.g. 100), we will print the error
            if batch_index % batch_index_check_training_loss == 0:
                # :>3 means 3 figures; :>4 means 4 figures; .3f means float with 3 decimals
                # TODO: batch_time * batch_index_check_training_loss is complaining about it's still a float?
                print(
                    'Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.
                    format(epoch, epochs, batch_index, ds.sub.num_questions_word2count // batch_size,
                           total_training_loss_error / batch_index_check_training_loss,
                           int(batch_time * batch_index_check_training_loss)))
                # Recompute total training loss error because we are done with 100 batches
                total_training_loss_error = 0

            # At every batch_index_check_validation_loss we reset total_validation_loss_error
            if batch_index % batch_index_check_validation_loss == 0 and batch_size > 0:
                total_validation_loss_error = 0
                starting_time = time.time()

                # TODO work on batch validation
                for batch_index_validation, (padded_question_in_batch, padded_answers_in_batch) in enumerate(
                        ds.get_validation_batches(batch_size)):
                    # Validation only contains new data that will be used for observations
                    # Probability is 1 when we are doing validation
                    batch_validation_loss_error = model.validate_batch(
                        inputs=padded_question_in_batch, targets=padded_answers_in_batch, learning_rate=learning_rate)

                    total_validation_loss_error += batch_validation_loss_error
                ending_time = time.time()
                batch_time = ending_time - starting_time
                average_validation_loss_error = total_validation_loss_error / (
                    len(ds.sub.validation_questions) / batch_size)
                print('Validation Loss Error: {:>6.3f}, '
                      'Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
                learning_rate *= learning_rate_decay

                # if lr goes below min_learning_rate
                if learning_rate < min_learning_rate:
                    learning_rate = min_learning_rate

                # Early stopping
                list_validation_loss_error.append(average_validation_loss_error)
                # If average_validation_loss_error is lower than every validation_loss_error_we got
                # do an early stopping
                if average_validation_loss_error <= min(list_validation_loss_error):
                    print('I speak better now!!')
                    # Sampling
                    sample_reply(model, ds)

                    early_stopping_check = 0
                    # Save weights
                    model.save_model(checkpoint)
                else:
                    print('Sorry I do not speak better, I need to practice more')
                    early_stopping_check += 1
                    if early_stopping_check == early_stopping_stop:
                        break
        if early_stopping_check == early_stopping_stop:
            print('My apologies, I cannot speak better anymore. This is the best I can do.')
            break


if __name__ == "__main__":
    main()
