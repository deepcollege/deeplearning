import os
import re
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
    CHARS_WHITE_LIST = r'[^a-z\s]'
    data_file_url = ('gameofthrones.txt', 'https://www.dropbox.com/s/p9086ydux9an8e7/gameofthrones.txt?dl=1')
    input_dir = '/input'

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
        text = re.sub(self.CHARS_WHITE_LIST, '', text)

        return text

    def maybe_download(self):
        fname, furl = self.data_file_url
        input_folder = '{input_dir}/got'.format(input_dir=self.input_dir)
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


if __name__ == "__main__":
    gotData = GOTData()
    gotData.maybe_download()
