import os
import io
import re
import pickle
import errno
import argparse
from .utils.file_helper import file_exists, try_create_dir
from urllib.request import urlopen
# from urllib2 import urlopen
from pprint import pprint
from sklearn.model_selection import train_test_split

from .utils.pprint_helper import Head

# List of NLP tokens
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']


def clean_text(text):
    """
    English helper to clean text
    :param text:
    :return:
    """
    text = text.lower()
    text = re.sub(r"i'm", 'i am', text)
    text = re.sub(r"he's", 'he is', text)
    text = re.sub(r"she's", 'she is', text)
    text = re.sub(r"that's", 'that is', text)
    text = re.sub(r"what's", 'what is', text)
    text = re.sub(r"where's", 'where is', text)
    text = re.sub(r"\'ll", ' will', text)
    text = re.sub(r"\'ve", ' have', text)
    text = re.sub(r"\'re", ' are', text)
    text = re.sub(r"\'d", ' would', text)
    text = re.sub(r"won't", 'will not', text)
    text = re.sub(r"can't", 'cannot', text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", '', text)
    return text


def convert_string2int(question, word2int):
    """
	Converting the raw string question to an integer representation
	:param question:
	:param word2int:
	:return:
	"""
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]


def convert_word_to_count(counter={}, doc=[]):
    """
    In-memory based simple word_to_count
    :param counter: Counter object to be returned
    :param doc: an entire document
    :return:
    """
    for sentence in doc:
        for word in sentence.split():
            if word not in counter:
                counter[word] = 1
            else:
                counter[word] += 1
    return counter


def save_file_data(name, obj, input_path='/inputs'):
    """
    Saving an object as a file using pickle
    :param name:
    :param obj:
    :return:
    """
    filename = '{}/{}.pkl'.format(input_path, name)
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:    # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename.format(name), 'wb+') as output:
        pickle.dump(obj, output)


def read_file_data(name, input_path='/inputs'):
    """
    Reading
    :param name:
    :return:
    """
    filename = '{}/{}.pkl'.format(input_path, name)
    with open(filename, 'rb') as input:
        return pickle.load(input)


def save_list_to_file(the_list, filepath):
    """ Write a list into a file """
    with open(filepath, 'w') as file_handler:
        for item in the_list:
            file_handler.write("{}\n".format(item))


cornell_file_urls = [('movie_lines.txt', 'https://www.dropbox.com/s/sljz3iejzfrwf5b/movie_lines.txt?dl=1'),
                     ('movie_conversations.txt',
                      'https://www.dropbox.com/s/nk0r6raow7xkr8b/movie_conversations.txt?dl=1')]


class Cornell:
    sorted_clean_questions = None
    sorted_clean_answers = None
    questions_words_2_counts = None
    answers_words_2_counts = None
    answers_counts_2_words = None
    num_questions_word2count = 0    # Number of total word2count in questions dict
    num_answers_word2count = 0    # Number of total word2count in answers dict

    # Train test split
    training_validation_split = 0
    training_questions = None
    training_answers = None
    validation_questions = None
    validation_answers = None

    # System
    input_dir = '/inputs'
    output_dir = '/output'

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def load_as_raw(self):
        """
        Load the dataset as file to a certain location
        1. questions list
        2. answers list
        3. questions vocabs list
        4. answers vocabs list
        """

        # Q & A
        questions, answers = self.get_questions_answers()

        # Get vocabs

        # Step 4: cleaning the questions
        pprint('---- Step 4 cleaning questions ----')

        clean_questions = []
        for question in questions:
            clean_questions.append(clean_text(question))

        pprint(clean_questions, stream=Head(5))
        print('\n\n')
        """
        Step 5: Clean the answers
        """

        pprint('---- Step 5 cleaning answers ----')
        clean_answers = []
        for answer in answers:
            clean_answers.append(clean_text(answer))

        pprint(clean_answers, stream=Head(5))
        print('\n\n')
        """
        Step 6: Creating a dictionary that maps each word to its number of occurences
        """

        word2count = {}
        pprint('------ Step 6: counting words in questions ----')

        word2count = convert_word_to_count(word2count, clean_questions)

        pprint(word2count, stream=Head(5))
        print('\n\n')
        """
        Step 7:
        For example, for a question: can we make this quick  roxanne korrine and andrew barrett are having an incredibly horrendous public break up on the quad  again
        It counts each word occurence such as "can" and accumulates the count into word2count dict
        """
        pprint('------ Step 6: counting words in answers ----')

        word2count = convert_word_to_count(word2count, clean_answers)

        pprint(word2count, stream=Head(5))
        print('\n\n')

        keys = ['<unk>', '<s>', '</s>']

        """
        Step 8: Creating word 2 int(count) by filtering words that are greater than the threshold
        """

        pprint(
            '------ Step 8: questions_vocabs filtered by threshold (>) ----')
        threshold_questions = 20
        questions_vocabs = [] + keys
        for word, count in word2count.items():
            if count >= threshold_questions:
                if not word in questions_vocabs:
                    questions_vocabs.append(word)

        pprint(questions_vocabs, stream=Head(5))
        print('\n\n')
        """
        Step 9: Same as step 8 but for answers
        """
        pprint(
            '------ Step 9: answers_vocabs filtered by threshold (>) ----')
        threshold_answers = 20
        answers_vocabs = [] + keys
        for word, count in word2count.items():
            if count >= threshold_answers:
                if not word in answers_vocabs:
                    answers_vocabs.append(word)

        pprint(answers_vocabs, stream=Head(5))

        return  questions, answers, questions_vocabs, answers_vocabs

    def download_if_not_exist(self):
        """
        Downloads required data file for cornell if it does not exists already
        :return:
        """
        for (fname, furl) in cornell_file_urls:
            # dir_path = os.path.dirname(os.path.realpath(__file__))
            input_folder = '{input_dir}/cornell'.format(input_dir=self.input_dir)
            full_dirname = input_folder
            full_fname = '/'.join([full_dirname, fname])
            if not file_exists(full_fname):
                remote_file = urlopen(furl)
                data = remote_file.read()
                remote_file.close()
                # Try creating the dir
                try_create_dir(full_dirname)
                print('download if not exist fname:', fname, 'url:', furl)
                # Write the file
                with open(full_fname, 'wb') as f:
                    f.write(data)

    def get_questions_answers(self):
        """
				:return: raw questions and answers
				"""
        print('Checking inputs cornell folder')
        print(os.listdir('{input_dir}/cornell'.format(input_dir=self.input_dir)))
        movie_line_path = '{input_dir}/cornell/movie_lines.txt'.format(input_dir=self.input_dir)
        lines = io.open(movie_line_path, encoding='utf8', errors='ignore').read().split('\n')
        conversation_path = '{input_dir}/cornell/movie_conversations.txt'.format(input_dir=self.input_dir)
        conversations = io.open(
            conversation_path, encoding='utf8', errors='ignore').read().split('\n')

        id2line = {}
        """
				Step 1: Creating a dict that maps each line to its id
				"""

        pprint('---- Step 1 creating a dict of id2lines ----')
        pprint('Original lines')
        pprint(lines, stream=Head(5))
        print('\n')

        for line in lines:
            _line = line.split(' +++$+++ ')
            if len(_line) == 5:
                id2line[_line[0]] = _line[4]

        pprint('processed')
        pprint(id2line, stream=Head(5))
        print('\n')
        print('Validating data')
        for line in lines[:5]:
            _line = line.split(' +++$+++ ')
            if len(_line) == 5:
                pprint(id2line[_line[0]])
        print('\n\n')
        """
				Step 2: Creating a list of all of the conversations
				"""

        conversations_ids = []
        pprint('----- Step 2 creating conversation ids ----')
        pprint('Original conversations:')
        pprint(conversations, stream=Head(5))
        print('\n')

        for conversation in conversations[:-1]:
            # Example: u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']
            _convo = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", '').replace(' ', '')
            conversations_ids.append(_convo.split(','))

        pprint('processed')
        pprint(conversations_ids, stream=Head(10))
        print('\n\n')
        """
				Step 3: Creates questions and answers sequence according to conversation_ids
				"""

        questions = []
        answers = []
        pprint('---- Step 3 constructing sequence ----')
        pprint(conversations_ids, stream=Head(5))
        for conversation_id in conversations_ids:
            for i in range(len(conversation_id) - 1):
                questions.append(id2line[conversation_id[i]])
                answers.append(id2line[conversation_id[i + 1]])

        print('\nChecking questions sequences')
        pprint(questions, stream=Head(5))
        print('\nChecking answers sequences')
        pprint(answers, stream=Head(5))
        print('\n\n')
        """
				Documents of questions and answers 1:1 mapped based on Cornell Movie Corpus Dialog
				"""
        return questions, answers

    def get_word2int(self, word):
        return self.questions_words_2_counts[word]


class Dataset:
    sub = None    # Dataset object
    type = 'cornell'
    output_dir = '/output'
    inputs_dir = '/inputs'

    def __init__(self, FLAGS):
        self.output_dir = FLAGS.output
        self.inputs_dir = FLAGS.input

    def load_as_files(self, lazy=True):
        """ Parent load as files class used globally """
        # Handle lazy load
        if lazy:
            try:
                questions = read_file_data('questions', self.inputs_dir)
                answers = read_file_data('answers', self.inputs_dir)
                questions_vocabs = read_file_data('questions_vocabs', self.inputs_dir)
                answers_vocabs = read_file_data('answers_vocabs', self.inputs_dir)
            except Exception as e:
                print('Failed to lazy load !', e)
                questions, answers, questions_vocabs, answers_vocabs = self.sub.load_as_raw()
                save_file_data('questions', questions, self.inputs_dir)
                save_file_data('answers', answers, self.inputs_dir)
                save_file_data('questions_vocabs', questions_vocabs, self.inputs_dir)
                save_file_data('answers_vocabs', answers_vocabs, self.inputs_dir)
        else:
            questions, answers, questions_vocabs, answers_vocabs = self.sub.load_as_raw()

        # It will always use vi as src en as output
        X_train, X_test, y_train, y_test = train_test_split(questions, answers, test_size=0.2, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        # Train
        save_list_to_file(X_train, '{}/train.from'.format(self.inputs_dir))
        save_list_to_file(y_train, '{}/train.to'.format(self.inputs_dir))

        # Valid
        save_list_to_file(X_valid, '{}/dev.from'.format(self.inputs_dir))
        save_list_to_file(y_valid, '{}/dev.to'.format(self.inputs_dir))

        # Test
        save_list_to_file(X_test, '{}/test.from'.format(self.inputs_dir))
        save_list_to_file(y_test, '{}/test.to'.format(self.inputs_dir))

        # vocab
        save_list_to_file(questions_vocabs, '{}/vocab.from'.format(self.inputs_dir))
        save_list_to_file(answers_vocabs, '{}/vocab.to'.format(self.inputs_dir))

    def load(self):
        if self.type == 'cornell':
            self.sub = Cornell(
                input_dir=self.inputs_dir,
                output_dir=self.output_dir
            )

        if self.sub:
            self.sub.download_if_not_exist()

            # Always load the data as file
            self.load_as_files()
        else:
            raise Exception('Invalid dataset!')


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

    ds = Dataset(FLAGS=FLAGS)
    ds.load()

if __name__ == "__main__":
    main()
