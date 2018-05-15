import numpy as np
import os
import pandas as pd
import random
import sys

if __name__ == "__main__":
	# parse command-line arguments
	if len(sys.argv) != 4:
		print "usage: python extract-gromacs-data.py [dir] [outfile] [num-samples]"
		sys.exit(1)

	INPUT_DIR = sys.argv[1]
	OUTFILE = sys.argv[2]
	NUM_SAMPLES = int(sys.argv[3])

	# get list of all subdirectories
	dirs = ["%s/%s" % (INPUT_DIR, dir) for dir in os.listdir(INPUT_DIR)]
	dirs = ["%s/training-data" % (dir) for dir in dirs if os.path.isdir(dir)]

	# initialize data matrix
	num_features = None
	data = None
	columns = []

	# initialize labels
	labels = []

	# iterate through each subdirectory
	count = 0
	for dir in dirs:
		print dir

		label = dir.split("/")[-2].split("-")[0]
		files = random.sample(os.listdir(dir), NUM_SAMPLES / len(dirs))

		# iterate through each sample file
		for f in files:
			# read sample file
			df = pd.read_csv("%s/%s" % (dir, f), sep="\t", header=None)

			# allocate data matrix on first sample
			if len(columns) == 0:
				num_features = len(df)
				data = np.empty((NUM_SAMPLES, num_features), dtype=np.float32)
				columns = df[2].values

			# HACK: some sample files have duplicate rows
			elif len(df) == num_features * 2:
				df = df.iloc[::2]

			# make sure features are ordered the same
			elif (columns != df[2].values).all():
				print "error: mismatched features"
				sys.exit(1)

			# append sample to data matrix
			data[count] = df[4]
			count += 1

		# append labels
		labels += [label for f in files]

	# save data matrix
	df = pd.DataFrame(data, None, columns)
	df.to_csv(OUTFILE, sep="\t")

	# TODO: save labels
	# print labels
