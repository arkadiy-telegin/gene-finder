from itertools import product
import pickle
from textwrap import fill
from shutil import copyfileobj
import numpy as np

train = 'datasets/train/'
test = 'datasets/test/'


class HMM:
    def __init__(self):
        """
        Z0: Non-coding, d=1
        Z1: Forward-start, d=3
        Z2: Forward-coding, d=3
        Z3: Forward-stop, d=3
        Z4: Reverse-stop, d=3
        Z5: Reverse-coding, d=3
        Z6: Reverse-start, d=3
        """
        self.pi = (1, 0, 0, 0, 0, 0, 0)  # Start probabilities.
        self.A = None  # Transition probabilities.
        self.phi = None  # Emission probabilities.
        self.counts_A = None  # Raw transition counts.
        self.counts_phi = None  # Raw emission counts.

    def count(self, data, labels):
        """Training by counting."""
        letters = 'AGCT'
        perms = [''.join(x) for x in product(letters, repeat=3)]
        self.counts_A = np.zeros((7, 7))
        self.counts_phi = {
            0: {key: 0 for key in letters},
            1: {key: 0 for key in perms},
            2: {key: 0 for key in perms},
            3: {key: 0 for key in perms},
            4: {key: 0 for key in perms},
            5: {key: 0 for key in perms},
            6: {key: 0 for key in perms},
        }

        for genome, annotation in zip(data, labels):
            g = open(genome, 'r')
            a = open(annotation, 'r')

            next_a = a.read(1)
            while True:

                if next_a == 'N':
                    self.counts_A[0, 0] += 1
                    next_g = g.read(1)
                    self.counts_phi[0][next_g] += 1

                if next_a == 'C':
                    self.counts_A[0, 1] += 1

                    # Read first triplet.
                    next_g = g.read(3)
                    self.counts_phi[1][next_g] += 1
                    assert a.read(2) == 'CC'

                    self.counts_A[1, 2] += 1
                    self.counts_A[2, 2] -= 1  # No double counting.
                    while (next_a := a.read(1)) == 'C':
                        self.counts_A[2, 2] += 1
                        next_g = g.read(3)
                        self.counts_phi[2][next_g] += 1
                        assert a.read(2) == 'CC'

                    if next_a == '':  # EOF
                        break

                    self.counts_A[2, 2] -= 1
                    self.counts_A[2, 3] += 1
                    self.counts_phi[2][next_g] -= 1
                    self.counts_phi[3][next_g] += 1

                    if next_a == 'N':
                        self.counts_A[3, 0] += 1
                        self.counts_A[0, 0] -= 1
                        continue

                    if next_a == 'R':
                        self.counts_A[3, 4] += 1
                        self.counts_A[0, 4] -= 1
                        continue

                if next_a == 'R':
                    self.counts_A[0, 4] += 1

                    next_g = g.read(3)
                    self.counts_phi[4][next_g] += 1
                    assert a.read(2) == 'RR'

                    self.counts_A[4, 5] += 1
                    self.counts_A[5, 5] -= 1
                    while (next_a := a.read(1)) == 'R':
                        self.counts_A[5, 5] += 1
                        next_g = g.read(3)
                        self.counts_phi[5][next_g] += 1
                        assert a.read(2) == 'RR'

                    if next_a == '':  # EOF
                        break

                    self.counts_A[5, 5] -= 1
                    self.counts_A[5, 6] += 1
                    self.counts_phi[5][next_g] -= 1
                    self.counts_phi[6][next_g] += 1

                    if next_a == 'N':
                        self.counts_A[6, 0] += 1
                        self.counts_A[0, 0] -= 1
                        continue

                    if next_a == 'C':
                        self.counts_A[6, 1] += 1
                        self.counts_A[0, 1] -= 1
                        continue

                next_a = a.read(1)
                if next_a == '':  # EOF
                    break

            g.close()
            a.close()

    def fit(self):
        self.A = np.zeros((7, 7))
        total_counts = [sum(self.counts_A[i,]) for i in range(7)]
        for i in range(7):
            for j in range(7):
                self.A[i, j] = self.counts_A[i, j] / total_counts[i]

        total_em = [sum(self.counts_phi[i].values()) for i in range(7)]
        self.phi = {
            state: {
                codon: self.counts_phi[state][codon] / total_em[state]
                for codon in self.counts_phi[state].keys()
            }
            for state in range(7)
        }

    def predict(self, genome):
        """
        Predict the annotation of genome using Viterbi decoding.
        """
        N = len(genome)
        k = 7
        w = np.full((k, N), -np.inf)
        d = [1] + [3] * 6

        w[0, 0] = np.log(self.phi[0].get(genome[0], 0))

        for i in range(1, N):
            for j in range(k):
                if i - d[j] < 0:
                    continue

                p_observed = np.log(sum([self.phi[j].get(genome[i - d[j] + 1 : i + 1], -np.inf)]))
                w[j, i] = np.max(w[:, i - d[j]] + np.log(self.A[:, j]) + p_observed)

        trans = 'N' + 'C' * 3 + 'R' * 3
        decoded = ''

        state = np.argmax(w[:, N - 1])
        last_state = state
        decoded += trans[state] * d[state]

        i = N - 1 - d[state]
        j = N - 1
        while i >= 0:
            state = np.argmax(w[:, i]
                                  + np.log(self.A[:, last_state])
                                  + np.log(self.phi[last_state].get(genome[i + 1 : j + 1], -np.inf)))
            decoded += trans[state] * d[state]
            j = i
            i -= d[state]
            last_state = state

        return decoded[::-1]

    def combine_params(self, **kwargs):
        letters = 'AGCT'
        perms = [''.join(x) for x in product(letters, repeat=3)]
        self.counts_A = np.zeros((7, 7))
        self.counts_phi = {
            0: {key: 0 for key in letters},
            1: {key: 0 for key in perms},
            2: {key: 0 for key in perms},
            3: {key: 0 for key in perms},
            4: {key: 0 for key in perms},
            5: {key: 0 for key in perms},
            6: {key: 0 for key in perms},
        }

        for filename in kwargs['A']:
            next_A = np.load(filename)
            self.counts_A += next_A

        for filename in kwargs['phi']:
            f = open(filename, 'rb')
            next_phi = pickle.load(f)
            f.close()

            for state, val in next_phi.items():
                for codon, counts in val.items():
                    self.counts_phi[state][codon] += counts


def read_fasta_file(filename):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.
    Lines starting with ';' in the FASTA file are ignored.
    """
    lines = []
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or line.startswith('>') or not line:
                continue
            else:
                lines.append(line)
    return ''.join(lines)


def read_unannotated():
    gens = []
    for i in range(6, 11):
        gens.append(read_fasta_file(f'datasets/test/genome{i}.fa'))

    return gens


def predict_unannotated():
    model = load_model()
    gens = read_unannotated()
    for i in range(6, 11):
        with open(f'datasets/predict/pred-ann{i}.fa', 'w') as f:
            f.write(f'>pred-ann{i}\n')
            pred = model.predict(gens[i - 6])
            f.write(fill(pred, width=60))
            f.write('\n')

    with open('datasets/predict/pred-ann6-10.fa', 'w') as f:
        for path in [f'datasets/predict/pred-ann{i}.fa' for i in range(6, 11)]:
            with open(path, 'r') as g:
                copyfileobj(g, f)


def cross_validate():
    for i in range(1, 6):
        ids = set(range(1, 6)).difference([i])
        model = HMM()
        As = [f'datasets/gen{j}_A_counts.npy' for j in ids]
        phis = [f'datasets/gen{j}_phi_counts.pickle' for j in ids]
        model.combine_params(A=As, phi=phis)
        model.fit()
        genome = read_fasta_file(f'datasets/train/genome{i}.fa')
        pred = model.predict(genome)
        with open(f'datasets/train/pred-ann{i}.fa', 'w') as f:
            f.write(f'>pred-ann{i}\n')
            f.write(fill(pred, width=60))
            f.write('\n')


def save_training():
    model = HMM()
    for i in range(1, 6):
        model.count(
            [train + f'genome{i}.fa_nonl.fa'], [train + f'true-ann{i}.fa_nonl.fa']
        )
        f = open(f'datasets/gen{i}_phi_counts.pickle', 'wb')
        pickle.dump(model.counts_phi, f)
        f.close()
        np.save(f'datasets/gen{i}_A_counts.npy', model.counts_A)

    model.count(
        [train + f'genome{i}.fa_nonl.fa' for i in range(1, 6)],
        [train + f'true-ann{i}.fa_nonl.fa' for i in range(1, 6)],
    )
    f = open(f'datasets/phi_counts.pickle', 'wb')
    pickle.dump(model.counts_phi, f)
    f.close()
    np.save(f'datasets/A_counts.npy', model.counts_A)

    model.fit()
    f = open(f'datasets/phi.pickle', 'wb')
    pickle.dump(model.phi, f)
    f.close()
    np.save(f'datasets/A.npy', model.A)

    return model


def load_model():
    model = HMM()
    with open('datasets/phi.pickle', 'rb') as f:
        model.phi = pickle.load(f)
    model.A = np.load('datasets/A.npy')
    return model


if __name__ == '__main__':
    predict_unannotated()

