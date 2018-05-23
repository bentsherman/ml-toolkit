import numpy as np
import os
import pandas as pd
import random
import sys

if __name__ == "__main__":
	# parse command-line arguments
	if len(sys.argv) != 4:
		print "usage: python extract-sid.py [dir] [outfile] [num-samples]"
		sys.exit(1)

	INPUT_DIR = sys.argv[1]
	OUTFILE = sys.argv[2]
	NUM_SAMPLES = int(sys.argv[3])

	# get list of all subdirectories
	dirs = ["%s/%s" % (INPUT_DIR, dir) for dir in os.listdir(INPUT_DIR)]
	dirs = ["%s/training-data" % (dir) for dir in dirs if os.path.isdir(dir)]

	# HACK: reduce sample size to multiple of directories
	NUM_SAMPLES = NUM_SAMPLES / len(dirs) * len(dirs)

	# iterate through each subdirectory
	num_features = 0
	X = None
	y = None
	count = 0

	for dir in dirs:
		print dir

		label = dir.split("/")[-2].split("-")[0]
		files = random.sample(os.listdir(dir), NUM_SAMPLES / len(dirs))

		# iterate through each sample file
		for f in files:
			# read sample file
			df = pd.read_csv("%s/%s" % (dir, f), sep="\t", header=None)

			# allocate X and y on first sample
			if num_features == 0:
				num_features = len(df)
				X = pd.DataFrame(np.empty((NUM_SAMPLES, num_features), dtype=np.float32), None, df[2].values)
				y = pd.DataFrame(np.empty((NUM_SAMPLES, 1), dtype=object), None, ["structure"])

			# HACK: some sample files have duplicate rows
			elif len(df) == num_features * 2:
				print "warning: file '%s/%s' has duplicate rows" % (dir, f)
				df = df.iloc[::2]

			# make sure features are ordered the same
			elif (X.columns != df[2].values).all():
				print "error: mismatched features"
				sys.exit(1)

			# append sample
			X.values[count] = df[4]
			y.values[count] = label
			count += 1

	# save data matrix
	df = X.join(y)
	df.to_csv(OUTFILE, sep="\t")
