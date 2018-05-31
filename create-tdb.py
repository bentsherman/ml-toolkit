import json
import numpy as np
import os
import pandas as pd
import random
import sys

if __name__ == "__main__":
	# parse command-line arguments
	if len(sys.argv) != 4:
		print("usage: python create-tdb.py [excel-file] [data-file] [config-file]")
		sys.exit(1)

	EXCELFILE = sys.argv[1]
	DATAFILE = sys.argv[2]
	CONFIGFILE = sys.argv[3]

	# load excel file
	df = pd.read_excel(EXCELFILE)

	# TODO: cleanup ?

	# save data matrix
	df.to_csv(DATAFILE, sep="\t")

	# initialize config object
	config = {
		"numerical": [],
		"categorical": [],
		"output": []
	}

	# add data frame columns to config
	for column in df.columns:
		if df[column].dtype == "object":
			config["categorical"].append(column)
		else:
			config["numerical"].append(column)

	# save config file
	configfile = open(CONFIGFILE, "w")
	json.dump(config, configfile, indent=2)
	configfile.close()
