

for size in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]:
	unique = set()
	total = 0

	with open("testTarget.txt") as f:
		for i, line in enumerate(f):
			if i < size:
				words = line.strip('\n').split()
				total += len(words)
				unique.update(words)

	print(len(unique) / total)
