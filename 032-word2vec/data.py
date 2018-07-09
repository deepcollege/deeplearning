import os
import re
import io
import collections
# from urllib.request import urlopen
from urllib2 import urlopen


# TODO: Come up with a generalised module structure...
def file_exists(path):
    """Test whether a path exists.  Returns False for broken symbolic links"""
    try:
        st = os.stat(path)
    except os.error:
        return False
    return True


def try_create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class GOTData:
    vocab_size = 0  # number of vocabs
    skip_gram_pairs = []  # list of skip gram pairs
    skip_gram_pairs_words = []

    # hparams
    window_size = None  # data window size
    clean_text_whitelist = None  # char whitelists during normalise_text
    data_file_url = None  # dataset location
    input_dir = None  # globally applicable input_dir path

    def __init__(self, **kwargs):
      self.window_size = kwargs.get('window_size', 1)
      self.clean_text_whitelist = kwargs.get('clean_text_whitelist', r'[^a-z\s]')
      self.data_file_url = kwargs.get('data_file_url',
                                      ('gameofthrones.txt', 'https://www.dropbox.com/s/p9086ydux9an8e7/gameofthrones.txt?dl=1'))
      self.input_dir = kwargs.get('input_dir', '/input/got')

    def normalise_text(self, text):
        '''
    Normalises a given text
    :param text:
    :return:
    '''
        text = text.lower()
        # Replacing some known words
        text = re.sub(r"i'm", 'i am', text)
        text = re.sub(r"he's", 'he is', text)
        text = re.sub(r"she's", 'she is', text)
        text = re.sub(r"that's", 'that is', text)
        text = re.sub(r"what's", 'what is', text)
        text = re.sub(r"where's", 'where is', text)

        # Replacing non-whitelisted chars with an empty char
        text = re.sub(self.clean_text_whitelist, '', text)

        return text

    def _maybe_download(self):
        fname, furl = self.data_file_url
        input_folder = '{input_dir}'.format(input_dir=self.input_dir)
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

    def load(self):
      self._maybe_download()
      fname, _ = self.data_file_url
      file_path = '{input_dir}/{fname}'.format(
        input_dir=self.input_dir,
        fname=fname)
      # Initial split, still contains the empty lines
      lines = io.open(file_path, encoding='utf8',
                      errors='ignore').read().split('\n')

      # Normalising each sentence
      normalised_sentences = []
      for line in lines:
        sentence = self.normalise_text(line)
        normalised_sentences.append(sentence)

      filtered_sentences = []
      for line in normalised_sentences:
        if line != u'':
          filtered_sentences.append(line)

      # sentences to words and count
      words = " ".join(filtered_sentences).split()
      count = collections.Counter(words).most_common()
      print('Word count', count[:5])

      # Build dictionaries
      unique_words = [i[0] for i in count]
      dic = {w: i for i, w in
             enumerate(unique_words)}  # dic, word -> id cats:0 dogs:1 ......

      self.vocab_size = len(dic)
      print('Vocab size:', self.vocab_size)

      # Make indexed word data
      data = [dic[word] for word in
              words]  # count rank for every word in words
      print('Sample data', data[:10], words[:10])

      # Let's make a training data for window size 1 for simplicity

      cbow_pairs = []
      for i in range(1, len(data) - self.window_size):
        cbow_pairs.append(
          [[data[i - self.window_size], data[i + self.window_size]], data[i]])

      print('Context pairs rank ids', cbow_pairs[:5])
      print()

      cbow_pairs_words = []
      for i in range(1, len(words) - self.window_size):
        cbow_pairs_words.append(
          [[words[i - self.window_size], words[i + self.window_size]], words[i]])
      print('Context pairs words', cbow_pairs_words[:5])

      # Creating the skip-gram
      self.skip_gram_pairs = []

      for c in cbow_pairs:
        self.skip_gram_pairs.append([c[1], c[0][0]])
        self.skip_gram_pairs.append([c[1], c[0][1]])
      print('skip-gram pairs', self.skip_gram_pairs[:5])

      self.skip_gram_pairs_words = []
      for c in cbow_pairs_words:
        self.skip_gram_pairs_words.append([c[1], c[0][0]])
        self.skip_gram_pairs_words.append([c[1], c[0][1]])
      print('skip-gram pairs words', self.skip_gram_pairs_words[:5])


if __name__ == "__main__":
    gotData = GOTData()
    gotData.load()