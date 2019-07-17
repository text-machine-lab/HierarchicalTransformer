"""Converts a single Twitter conversation corpus to two files in machine translation format. From Github:
https://github.com/Phylliida/Dialogue-Datasets/blob/master/TwitterLowerAsciiCorpus.txt"""
from nltk import word_tokenize
import random
import os

class TwitterConversationCorpusConverter:
    """This class takes a single file of Twitter conversations as input. Eac conversation consists of
    utterances each separated by newline. Conversations are separated by four newlines. This class converts
    this format into two files in machine translation style format. The first file contains conversation histories
    and the second contains responses. The files are aligned."""
    def __init__(self):
        self.sep = '<sep>'

    def load_examples(self, conv_file):
        """Loads examples from Twitter corpus text file.

        :param conv_file string containing path to file with Twitter conversations

        :return list of tuples. Each tuple contains two strings (history, response). History contains <sep> token
        between each pair of utterances in the conversation"""
        with open(conv_file, 'r') as f:
            doc = f.read()

        convos = doc.split('\n\n\n\n')

        examples = []
        for convo in convos:
            convo = convo.strip()
            if convo != '':
                utters = convo.split('\n')
                clean_utters = [clean_utterance(utter) for utter in utters]
                for i in range(len(clean_utters)):
                    history = (' ' + self.sep + ' ').join(clean_utters[:i])
                    response = clean_utters[i]
                    examples.append((history, response))
        random.shuffle(examples)

        return examples

    def export_examples(self, examples, hist_file, resp_file):
        """Exports examples to aligned history and response files in machine translation format."""

        with open(hist_file, 'w') as h:
            with open(resp_file, 'w') as r:
                for example in examples:
                    h.write(example[0] + '\n')
                    r.write(example[1] + '\n')


def clean_utterance(text):
    #TODO tokenize with nltk and lowercase among other things
    text_tk = ' '.join(word_tokenize(text.lower().strip()))
    return text_tk


if __name__ == '__main__':
    twitter_file = 'twitter_conv_corpus.txt'
    history_file = 'histories.txt'
    response_file = 'responses.txt'
    output_dir = 'twitter/'
    ds = TwitterConversationCorpusConverter()

    examples = ds.load_examples(twitter_file)

    print('Loaded examples from file: %s' % twitter_file)
    print(len(examples))
    print(examples[0])
    os.makedirs(output_dir, exist_ok=True)
    ds.export_examples(examples, output_dir + history_file, output_dir + response_file)
    print('Saved output in %s and %s files' % (history_file, response_file))