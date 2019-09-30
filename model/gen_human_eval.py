"""This script will load model samples and produce a human evaluation file."""
import os
import random
import pickle
import sys

EVAL_DIR = '../data/model_samples'
OUTPUT_FILE = '../data/human_evaluation.txt'
ANSWER_FILE = '../data/human_answer_key.pkl'


# the key model is the model we will compare all other models to
KEY_MODEL = 'unet'
EXAMPLES_PER_MODEL = 10000
MAX_LENGTH = 150
SEED = 1234

def load_examples(filename):
    raw = open(os.path.join(EVAL_DIR, filename), 'r').read()
    # split examples by gap
    examples = raw.split('\n\n')
    # split each example into input sentence, ground truth, and generated response
    examples = [example.split('\n') for example in examples]
    # remove epoch indicators from the data
    examples = [example for example in examples if len(example) > 1]
    # check that tuple contains the required data
    final_examples = []
    for example in examples:
        assert 'Input sentence: ' in example[0]
        assert 'Ground truth: ' in example[1]
        assert 'Generated response: ' in example[2]
        final_examples.append([example[0].replace('Input sentence: ', ''),
                               example[1].replace('Ground truth: ', ''),
                               example[2].replace('Generated response: ', '')])

    return final_examples


def extract_examples_per_model(key_model_name, eval_dir, max_examples, verbose=True):
    # grab all filenames in evaluation directory
    model_files = [f for f in os.listdir(eval_dir) if os.path.isfile(os.path.join(eval_dir, f))]
    model_names = [f.split('_')[0] for f in model_files]
    model_examples = [load_examples(f) for f in model_files]

    # prune model examples
    model_examples = [examples[:max_examples] for examples in model_examples]

    # isolate key model from models
    key_idx = model_names.index(key_model_name)
    key_model_file = model_files[key_idx]
    key_model_examples = model_examples[key_idx]
    model_names.remove(key_model_name)
    model_files.remove(key_model_file)
    model_examples.remove(key_model_examples)

    if verbose:
        print('Model names: %s' % str(model_names))
        print('Model files: %s' % str(model_files))
        print('Key name: %s' % key_model_name)
        print('Key file: %s' % key_model_file)

    return model_examples, key_model_examples, model_names


def shuffle_tuple(x):
    """x is a tuple (a, b), we swap a and b with 50% probability"""
    a, b = x
    if random.randint(0, 1) == 1:
        return (a, b)
    else:
        return (b, a)


def compare_model_examples(key_model_name, model_examples, key_model_examples,  model_names):
    model_pairs = [zip(key_model_examples, examples) for examples in model_examples]
    pairs_flat = [((key_model_name, pair[0]), (model_name, pair[1])) for model_name, pairs in zip(model_names, model_pairs)
                  for pair in pairs]
    return pairs_flat


if __name__ == '__main__':

    # predictable behavior
    random.seed(SEED)

    model_examples, key_model_examples, model_names = extract_examples_per_model(KEY_MODEL, EVAL_DIR, EXAMPLES_PER_MODEL)

    for name, examples in zip(model_names, model_examples):
        print('%s: %s examples' % (name, len(examples)))

    pairs_flat = compare_model_examples(KEY_MODEL, model_examples, key_model_examples, model_names)

    # we shuffle each pair for each example, then shuffle all examples
    pairs_flat = [shuffle_tuple(pair) for pair in pairs_flat]
    random.shuffle(pairs_flat)


    # form (history, unet_response, baseline_response) tuples
    # write tuples to file for humans to judge the best response


    # remove all pairs which have different histories
    pairs_flat_same_history = [pair for pair in pairs_flat if pair[0][1][0] == pair[1][1][0]]

    print('Fraction pairs with different histories: %s' % (1 - len(pairs_flat_same_history)/len(pairs_flat)))

    pairs_flat_no_pruned_history = [pair for pair in pairs_flat_same_history if len(pair[0][1][0].split()) < MAX_LENGTH]

    print('Fraction pairs with pruned histories: %s' % (1 - len(pairs_flat_no_pruned_history)/len(pairs_flat_same_history)))

    # we only use pairs which have the same history and that history is not pruned to max length
    pairs_flat = pairs_flat_no_pruned_history

    # write answer key to file, make sure both files are aligned
    pickle.dump(pairs_flat, open(ANSWER_FILE, 'wb'))

    print(pairs_flat[1])
    print('Number of comparisons: %s' % len(pairs_flat))

    with open(OUTPUT_FILE, 'w') as f:
        for pair in pairs_flat:

            ((first_name, first_example), (second_name, second_example)) = pair

            first_history, first_label, first_response = first_example
            second_history, second_label, second_response = second_example

            other_model = first_name if first_name != KEY_MODEL else second_name

            better = 'left' if first_name != KEY_MODEL else 'right'

            f.write('###\n')
            f.write('H: %s\n' % first_history)
            f.write('R #1: %s\n' % first_response)
            f.write('R #2: %s\n' % second_response)
            f.write('Which is better?: \n')

    # assert evaluation file has same number of pairs as answer key
    assert len(pairs_flat) == len(open(OUTPUT_FILE, 'r').read().split('###')) - 1



