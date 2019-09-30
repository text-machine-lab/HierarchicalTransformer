from model.gen_human_eval import load_examples
import os

EVAL_DIR = '../data/personachat_model_samples'
KEY_MODEL = 'unet'
EXAMPLES_PER_MODEL = 10000
PAGE_BREAK = 1000
MAX_LENGTH = 150
SEED = 1234
BIGRAM = False

if __name__ == '__main__':
    model_files = [f for f in os.listdir(EVAL_DIR) if os.path.isfile(os.path.join(EVAL_DIR, f))]
    model_names = [f.split('_')[0] for f in model_files]
    model_examples = [load_examples(os.path.join(EVAL_DIR, f)) for f in model_files]

    for name, examples in zip(model_names, model_examples):
        token_counts = {}
        print('Model name: %s' % name)

        # grab generated responses
        responses = [example[2] for example in examples]
        print('Number of responses: %s' % len(responses))

        for response in responses:
            prev_token = None
            for token in response.split():
                if prev_token is not None:
                    if BIGRAM:
                        token = (prev_token, token)
                    if token not in token_counts:
                        token_counts[token] = 1
                    else:
                        token_counts[token] += 1

                prev_token = token

        # divide number of tokens by total count to get fraction of unique 1-grams
        n_unique_tokens = len(token_counts)
        n_tokens = sum([token_counts[token] for token in token_counts])

        print('Number of tokens: %s' % n_tokens)
        print('Number of unique tokens: %s' % n_unique_tokens)
        print('Fraction of unique tokens per response: %s' % (n_unique_tokens / len(responses)))

