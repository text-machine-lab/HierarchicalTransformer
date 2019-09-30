import pickle
import os

KEY_MODEL = 'unet'

EVAL_DIR = '../data/model_samples'
OUTPUT_FILE = '../data/human_evaluation.txt'
ANSWER_FILE = '../data/human_answer_key.pkl'

pairs = pickle.load(open(ANSWER_FILE, 'rb'))

human_evals = open(OUTPUT_FILE, 'r').read().split('###')[1:]

assert len(pairs) == len(human_evals)

model_files = [f for f in os.listdir(EVAL_DIR) if os.path.isfile(os.path.join(EVAL_DIR, f))]
model_names = [f.split('_')[0] for f in model_files]
model_names.remove(KEY_MODEL)

key_wins   = {model_name: 0 for model_name in model_names}
key_ties   = {model_name: 0 for model_name in model_names}
key_losses = {model_name: 0 for model_name in model_names}


for answer, eval in zip(pairs, human_evals):

    ((first_name, first_example), (second_name, second_example)) = answer

    _, history, first_response, second_response, vote, _ = eval.split('\n')
    history = history.replace('H:', '').strip()
    first_response = first_response.replace('R #1:', '').strip()
    second_response = second_response.replace('R #2:', '').strip()
    vote = vote.replace('Which is better?:', '').strip().lower()

    # print(history)
    # print(first_response)
    # print(second_response)
    # print(vote)

    # either the l
    other_model = first_name if first_name != KEY_MODEL else second_name


    first_is_key = True if first_name == KEY_MODEL else False

    if vote != '':
        if vote == 'left' or vote == 'first' or vote == '1' or vote == '#1':
            # user voted that first response was better
            if first_is_key:
                key_wins[other_model] += 1
            else:
                key_losses[other_model] += 1
        elif vote == 'both' or vote == 'neither' or vote == 'same' or vote == 'tie' or vote == '0':
            # user voted that both responses were the same
            key_ties[other_model] += 1
        elif vote == 'right' or vote == 'second' or vote == '2' or vote == '#2':
            # user voted that second response was better
            if first_is_key:
                key_losses[other_model] += 1
            else:
                key_wins[other_model] += 1
        else:
            raise ValueError('Used incorrect vote: %s' % vote)


for model_name, _, _ in zip(key_wins, key_ties, key_losses) :
    print('%s versus %s' % (KEY_MODEL, model_name))
    print('wins:', key_wins[model_name], ', ties:', key_ties[model_name], ', losses:', key_losses[model_name])