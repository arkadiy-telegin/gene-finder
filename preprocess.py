train = 'datasets/train/'
test = 'datasets/test/'


def remove_newline(file):
	f = open(file, 'r')
	f.readline()
	f.readline()
	genome = f.read()
	no_nl = genome.replace('\n', '')
	new_f = open(file + '_nonl.fa', 'w')
	new_f.write(no_nl)


if __name__ == '__main__':
	for i in range(1, 6):
		remove_newline(train + f'genome{i}.fa')
		remove_newline(train + f'true-ann{i}.fa')