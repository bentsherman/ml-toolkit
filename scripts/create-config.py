import json
import pandas as pd
import sys



if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("usage: python create-config.py [infile] [outfile]")
		sys.exit(1)

	# load data frame
	df = pd.read_csv(sys.argv[1], sep="\t")

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
	outfile = open(sys.argv[2], "w")
	json.dump(config, outfile, indent=2)
	outfile.close()
