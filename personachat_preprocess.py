import argparse
import os
import random
from tqdm import tqdm
import pickle
from model.utils import Vocab

eos = '<eos>'
pad = '<pad>'

def create_dialogues(filename, max_len):
    """
    partitions the lines of the movie into dialogues based on the time
        the lines happen, and the maximum length

    :param filename:
    :param max_len(int): the maximum number of tokens in the history
    :param max_time_interval(int): the maximum time between lines in a dialogue
    :param movie_id(int): the id of the movie
    :return:
        List: [(movie id, history, response)]
    """
    chat_id = 0
    history = list()
    chat_ids = list()
    response = list()
    dialogue = list()

    conversations = []

    with open(filename, 'r') as f:
        for line in f:

            # check if this is the start of a new dialogue
            if line[0] == '1':
                conversations.append(dialogue)
                dialogue = list()
                chat_id += 1

            # add current line in file to growing dialogue

            first_line = (line.split("\t")[0][1:].strip()).split(' ')
            second_line = (line.split("\t")[1].strip()).split(' ')

            first_line = first_line[:max_len-1] + [eos]
            second_line = second_line[:max_len-1] + [eos]


            first_line = first_line + [pad] * (max_len - len(first_line))
            second_line = second_line + [pad] * (max_len - len(second_line))

            dialogue.append(first_line)
            dialogue.append(second_line)

    return conversations[1:]

def main():
    """
    here is the plan: for each dialogue create a history sequence of sentences
    seperated by <s>. The sentences in the history must occur in a short time
    span from another so they are relevant. The last sentence becomes the response
    where the response must also be in the span
    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset_dir",
                        default="./datasets/personachat/raw",
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the xml for the task.")
    parser.add_argument("-output_dir",
                        default="./datasets/personachat/",
                        type=str,
                        required=False,
                        help="The output data dir.")
    parser.add_argument("-type",
                        default="none_original",
                        type=str,
                        required=False,
                        help="The genres you would like to use.")
    parser.add_argument("-max_sentence_tokens",
                        default=30,
                        type=int,
                        help="the maximum amout of sentence tokens")
    parser.add_argument("-a_nice_note",
                        default="only dialogues 1-10",
                        type=str,
                        required=False,
                        help="leave a nice lil note for yourself in the future")

    parser.add_argument('-train_split',
                        default=0.9,
                        type=float,
                        help='fraction of dataset to use for training, remainder is halved for val & test')

    parser.add_argument('-vocab_size',
                        default=20000,
                        type=int,
                        help='maximum size of the vocabulary for training')

    args = parser.parse_args()

    filename = os.path.join(args.dataset_dir, "train_{}.txt".format(args.type))

    conversations = create_dialogues(filename, args.max_sentence_tokens)

    for conversation in conversations:
        for utterance in conversation:
            if len(utterance) != args.max_sentence_tokens:
                print('Length of utterance not equal max: %s' % len(utterance))
                exit()

    print(conversations[0])

    # shuffle dataset

    random.seed('seed')
    random.shuffle(conversations)

    print('Number of conversations: %s' % len(conversations))

    mean_n_convos = sum([len(conv) for conv in conversations]) / len(conversations)
    print('Average utterances per conversations: %s' % mean_n_convos)

    # this is format needed to train dialogue models on this domain
    def format_for_dialogue(conversations):
        conversation_length = [len(conv) for conv in conversations]
        sentence_length = [[sum([1 for token in sent if token != '<pad>']) for sent in conv] for conv in conversations]
        sentences = conversations
        return conversation_length, sentence_length, sentences

    val_idx = int(len(conversations) * args.train_split)
    test_idx = (len(conversations) + val_idx) // 2
    print(val_idx)

    train_convos = conversations[:val_idx]
    val_convos = conversations[val_idx:test_idx]
    test_convos = conversations[test_idx:]

    # construct vocab
    vocab = Vocab()
    vocab.add_dataframe(train_convos, tokenized=True)
    vocab.update(args.vocab_size)
    print('Vocab size: %s' % len(vocab))

    word2id_path = os.path.join(args.output_dir, 'word2id.pkl')
    id2word_path = os.path.join(args.output_dir, 'id2word.pkl')
    vocab.pickle(word2id_path, id2word_path)

    print('Split: train %s, val %s, test %s' % (len(train_convos), len(val_convos), len(test_convos)))

    os.makedirs(args.output_dir, exist_ok=True)

    train_convo_len, train_sent_len, train_sent = format_for_dialogue(train_convos)
    print('Example data')
    print(train_convo_len[0])
    print(train_sent_len[0])
    print(train_sent[0])
    print()

    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    pickle.dump(train_convo_len, open(os.path.join(args.output_dir, 'train', 'conversation_length.pkl'), 'wb'))
    pickle.dump(train_sent_len, open(os.path.join(args.output_dir, 'train', 'sentence_length.pkl'), 'wb'))
    pickle.dump(train_sent, open(os.path.join(args.output_dir, 'train', 'sentences.pkl'), 'wb'))

    val_convo_len, val_sent_len, val_sent = format_for_dialogue(val_convos)
    os.makedirs(os.path.join(args.output_dir, 'valid'), exist_ok=True)
    pickle.dump(val_convo_len, open(os.path.join(args.output_dir, 'valid', 'conversation_length.pkl'), 'wb'))
    pickle.dump(val_sent_len, open(os.path.join(args.output_dir, 'valid', 'sentence_length.pkl'), 'wb'))
    pickle.dump(val_sent, open(os.path.join(args.output_dir, 'valid', 'sentences.pkl'), 'wb'))

    test_convo_len, test_sent_len, test_sent = format_for_dialogue(test_convos)
    os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)
    pickle.dump(test_convo_len, open(os.path.join(args.output_dir, 'test', 'conversation_length.pkl'), 'wb'))
    pickle.dump(test_sent_len, open(os.path.join(args.output_dir, 'test', 'sentence_length.pkl'), 'wb'))
    pickle.dump(test_sent, open(os.path.join(args.output_dir, 'test', 'sentences.pkl'), 'wb'))



    # with open(filename, 'r') as f:
    #     lines = f.read()
    #
    # lines = lines.split('\n')
    #
    # lines = lines[:1000]
    #
    # lines = [line.replace('\t', ' <sep> ') for line in lines]
    #
    # lines = [line.replace('|',  ' <eoc> ') for line in lines]
    #
    # tokens = [token for line in lines for token in line.split(' ')[1:]]
    #
    # text = ' '.join(tokens)
    #
    # convos = text.split(' <eoc> ')
    #
    # random.shuffle(convos)
    #
    # for i in range(5):
    #     print(convos[i])
    #     print()

    # # split each line by tabs, each tab separating two utterances
    # print('Split utterances in each line')
    # lines = [line.split('\t') for line in tqdm(lines)]
    # # split each utterance into tokens (list of list of tokens)
    # print('Split tokens in each utterance')
    # utterances = [utterance.split(' ') for line in tqdm(lines) for utterance in line]
    # # remove all number signifiers and flatten
    # print('Flatten lines into one string of utterances')
    # text = ' '.join([token for utterance in tqdm(utterances) for token in utterance if not token.isdigit()])
    #
    # print('Split utterances into conversations')
    # conversation = text.split('|')

    # print(conversation[0])

if __name__ == '__main__':
    main()