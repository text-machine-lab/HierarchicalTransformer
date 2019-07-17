import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-data', required=True)

opt = parser.parse_args()

lens = []

with open(opt.data, 'r') as f:
    for line in f:
        tokens = line.split()
        lens.append(len(tokens))

mean = np.mean(lens)
std = np.std(lens)

print('# tokens mean: %s' % mean)
print('# tokens std: %s' % std)


