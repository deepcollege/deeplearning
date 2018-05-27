from tensorflow import tf
from .model import Seq2Seq


# Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index:start_index + batch_size]
        answers_in_batch = answers[start_index:start_index + batch_size]
        padded_questions_in_batch = np.array(
            apply_padding(questions_in_batch, questions_words_2_ints))
        padded_answers_in_batch = np.array(
            apply_padding(answers_in_batch, answers_words_2_ints))
        yield padded_questions_in_batch, padded_answers_in_batch


def main():
  for epoch in range(1, epochs + 1):
    for batch_index, (padded_question_in_batch,
                      padded_answers_in_batch) in enumerate(
      split_into_batches(training_questions,
                         training_answers, batch_size)):
  pass


if __name__ == "__main__":
    main()
