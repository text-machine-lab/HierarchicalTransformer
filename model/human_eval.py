"""This script will load models and produce a human evaluation file."""
import os
import random
import pickle

EVAL_DIR = '../data/model_samples'
OUTPUT_FILE = '../data/human_evaluation.txt'
ANSWER_FILE = '../data/human_answer_key.pkl'


# the key model is the model we will compare all other models to
KEY_MODEL = 'unet'
EXAMPLES_PER_MODEL = 1000
SEED = 1234

random.seed(SEED)

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

# grab all filenames in evaluation directory
model_files = [f for f in os.listdir(EVAL_DIR) if os.path.isfile(os.path.join(EVAL_DIR, f))]
model_names = [f.split('_')[0] for f in model_files]
model_examples = [load_examples(f) for f in model_files]

# prune model examples
model_examples = [examples[:EXAMPLES_PER_MODEL] for examples in model_examples]

# isolate key model from models
key_idx = model_names.index(KEY_MODEL)
key_model_file = model_files[key_idx]
key_model_examples = model_examples[key_idx]
model_names.remove(KEY_MODEL)
model_files.remove(key_model_file)
model_examples.remove(key_model_examples)

print('Model names: %s' % str(model_names))
print('Model files: %s' % str(model_files))
print('Key name: %s' % KEY_MODEL)
print('Key file: %s' % key_model_file)

# compare unet examples to each model

model_pairs = [zip(key_model_examples, examples) for examples in model_examples]

pairs_flat = [((KEY_MODEL, pair[0]), (model_name, pair[1])) for model_name, pairs in zip(model_names, model_pairs) for pair in pairs]

# form (history, unet_response, baseline_response) tuples
print(pairs_flat[1])
print('Number of comparisons: %s' % len(pairs_flat))
# write tuples to file for humans to judge the best response

# write answer key to file, make sure both files are aligned

pickle.dump(pairs_flat, pairs_flat)

with open(OUTPUT_FILE, 'w') as f:
    for pair in pairs_flat:

        ((key_name, key_example), (model_name, baseline_example)) = pair

        key_history, _, key_response = key_example
        base_history, _, base_response = baseline_example

        if key_history == base_history:
            pass
