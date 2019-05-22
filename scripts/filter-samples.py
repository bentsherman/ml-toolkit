import json
import pandas as pd
import sys



if __name__ == "__main__":
	if len(sys.argv) != 4:
		print("usage: python filter-samples.py [dataframe] [config] [outfile]")
		sys.exit(1)

	# load dataframe, config
	df = pd.read_csv(sys.argv[1], sep="\t")
	config = json.load(open(sys.argv[2]))

	# remove samples with high-variance output
	output_sd = "%s_SD" % config["output"][0]
	if output_sd in df.columns:
		mask = df[output_sd] < 1.5
		df = df[mask]

	# save dataframe
	df.to_csv(sys.argv[3], sep="\t")
