import json
import numpy as np
import os
import pandas as pd
import random
import sys

if __name__ == "__main__":
	# parse command-line arguments
	if len(sys.argv) != 5:
		print("usage: python create-sid.py [dir] [num-samples] [data-file] [config-file]")
		sys.exit(1)

	INPUT_DIR = sys.argv[1]
	NUM_SAMPLES = int(sys.argv[2])
	DATAFILE = sys.argv[3]
	CONFIGFILE = sys.argv[4]

	# get list of all subdirectories
	dirs = [d for d in os.listdir(INPUT_DIR) if os.path.isdir("%s/%s" % (INPUT_DIR, d))]
	dirs = ["%s/training-data" % (d) for d in dirs]

	# get list of all classes
	classes = [d.split("-")[0] for d in dirs]
	classes = list(set(classes))

	# iterate through each class
	num_features = 0
	X = None
	y = None
	count = 0

	for i in range(len(classes)):
		print(classes[i])

		# get list of all files in class
		class_dirs = ["%s/%s" % (INPUT_DIR, d) for d in dirs if d.split("-")[0] == classes[i]]
		files = sum([["%s/%s" % (d, f) for f in os.listdir(d)] for d in class_dirs], [])

		# sample files from list
		k = int(NUM_SAMPLES * (i + 1) / len(classes) - NUM_SAMPLES * i / len(classes))
		files = random.sample(files, k)

		# append each sample to data frame
		for f in files:
			# read sample file
			sample = pd.read_table(f, header=None)

			# allocate X and y on first sample
			if num_features == 0:
				num_features = len(sample)
				X = pd.DataFrame(np.empty((NUM_SAMPLES, num_features), dtype=np.float32), None, sample[2].values)
				y = pd.DataFrame(np.empty((NUM_SAMPLES, 1), dtype=object), None, ["structure"])

			# HACK: some sample files have duplicate rows
			elif len(sample) == num_features * 2:
				print("warning: file '%s' has duplicate rows" % (f))
				sample = sample.iloc[::2]

			# make sure features are ordered the same
			elif (X.columns != sample[2].values).all():
				print("error: mismatched features")
				sys.exit(1)

			# append sample
			X.values[count] = sample[4]
			y.values[count] = classes[i]
			count += 1

	# save data matrix
	df = X.join(y)
	df.to_csv(DATAFILE, sep="\t")

	# initialize config object
	config = {
		"numerical": X.columns.tolist(),
		"categorical": [],
		"output": y.columns.tolist()
	}

	# save config file
	configfile = open(CONFIGFILE, "w")
	json.dump(config, configfile, indent=2)
	configfile.close()
