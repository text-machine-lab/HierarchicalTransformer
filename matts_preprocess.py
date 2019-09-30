import csv
import argparse
import re
import os
import string
import json
from tqdm import tqdm

SEPERATOR_TOKEN = "<s>"

def get_history_response(dialogue):
    sentences = [" ".join(d) for d in dialogue]
    hisory = " <s> ".join(sentences[:-1])
    hisory = "<cls> " + hisory
    return hisory, "<cls> " +sentences[-1]


def get_line(soup):
    non_tokens = {'"', '\"', '\'', '\' ', '-'}
    line = list()

    for token in soup.find_all("w"):
        valid = 1
        text = token.text
        text = text.lower()
        if text == 're': text = 'are'
        if text == 's': text = 'is'
        if text == 'm': text = 'am'
        if text == 't': text = 'not'
        if text == 've': text = 'have'
        if text == 'il': text = 'will'
        text = text.replace('n\'', '')
        text = text.replace('\' ', '')
        text = text.replace('\'', '')
        text = text.replace('"', '')
        text = text.replace('\"', '')
        if text.isdigit():
            line.append("<NUM>")
            continue
        if not text in non_tokens:
            line.append(text)
    return line

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
    import pdb; pdb.set_trace()
    chat_id = 0
    history = list()
    chat_ids = list()
    response = list()
    dialogue = list()

    with open(filename, 'r') as f:
        for line in f:

            # check if this is the start of a new dialogue
            if line[0] == '1':
                dialogue = list()
                chat_id += 1

            # add current line in file to growing dialogue
            dialogue.append((line.split("\t")[0][1:].strip()).split(' '))

            # trim dialogue to maximum length
            while (sum([len(sentence) for sentence in dialogue[:-1]]) + len(
                    dialogue) - 1 > max_len):
                dialogue = dialogue[1:]

            # for every response in the conversation, treat previous responses as history and add conversation
            if len(dialogue) >= 2:
                h, r = get_history_response(dialogue)
                history.append(h)
                response.append(r)
                chat_ids.append(chat_id)

            
            dialogue.append((line.split("\t")[1].strip()).split(' '))
            while (sum([len(sentence) for sentence in dialogue[:-1]]) + len(
                    dialogue) - 1 > max_len):
                dialogue = dialogue[1:]

            h, r = get_history_response(dialogue)
            history.append(h)
            response.append(r)
            chat_ids.append(chat_id)
    return [(c, h, r) for c, h, r in zip(chat_ids, history, response)]


def main():
    """
    here is the plan: for each dialogue create a history sequence of sentences
    seperated by <s>. The sentences in the history must occur in a short time
    span from another so they are relevant. The last sentence becomes the response
    where the response must also be in the span
    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset_filename",
                        default="./datasets/personachat/raw",
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the xml for the task.")
    parser.add_argument("-output_dir",
                        default="./datasets/personachat/processed",
                        type=str,
                        required=False,
                        help="The output data dir.")
    parser.add_argument("-type",
                        default="none_original",
                        type=str,
                        required=False,
                        help="The genres you would like to use.")
    parser.add_argument("-max_history_tokens",
                        default=100,
                        type=int,
                        help="the maximum amout of history tokens")
    parser.add_argument("-a_nice_note",
                        default="only dialogues 1-10",
                        type=str,
                        required=False,
                        help="leave a nice lil note for yourself in the future")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    meta_data = dict()
    meta_data["args"] = dict()

    for arg in vars(args):
        meta_data["args"][arg] = getattr(args, arg)

    train_set = create_dialogues(os.path.join(args.dataset_filename,
                                              "train_{}.txt".format(args.type)),
                                 args.max_history_tokens)

    val_set = create_dialogues(os.path.join(args.dataset_filename,
                                            "valid_{}.txt".format(args.type)),
                               args.max_history_tokens)
    test_set = create_dialogues(os.path.join(args.dataset_filename,
                                            "test_{}.txt".format(args.type)),
                               args.max_history_tokens)

    meta_data["train_len"] = len(train_set)
    meta_data["val_len"] = len(val_set)
    meta_data["test_len"] = len(test_set)

    with open(os.path.join(args.output_dir, "meta_data"), 'w') as fp:
        json.dump(meta_data, fp, indent=4, sort_keys=True)

    with open(os.path.join(args.output_dir, "train.csv"), 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['movie_id', 'history', 'response'])
        for row in train_set:
            csv_out.writerow(row)

    with open(os.path.join(args.output_dir, "val.csv"), 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['movie_id', 'history', 'response'])
        for row in val_set:
            csv_out.writerow(row)

    with open(os.path.join(args.output_dir, "test.csv"), 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['movie_id', 'history', 'response'])
        for row in test_set:
            csv_out.writerow(row)

if __name__ == '__main__':
    main()