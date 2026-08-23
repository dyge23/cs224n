import utils

with open('../birth_dev.tsv', encoding='utf-8') as f:
    n = len(f.readlines())

total, correct = utils.evaluate_places('../birth_dev.tsv', ['London'] * n)
print(correct, total, 100 * correct / total)