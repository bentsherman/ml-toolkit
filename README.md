# ml-tools

This repository is a collection of command-line tools for data analysis and visualization. These tools essentially form a thin interface around several Python libraries including pandas, scikit-learn, and matplotlib / seaborn.

There are four primary scripts:

1. `classifiy.py`: classification algorithms
2. `cluster.py`: clustering algorithms
3. `regress.py`: regression algorithms
4. `visualize.py`: data visualization

Each script takes two inputs: (1) a tab-delimited data matrix and (2) a JSON configuration file. The data matrix is read as a pandas DataFrame; it should contain row-wise samples and should include both features and outputs. The JSON config file should specify numerical features, categorical features, and outputs. The following example could be for a dataset of housing prices:
```
{
   "numerical": [
      "age",
      "area",
   ],
   "categorical": [
      "state",
      "zip",
      "color",
      "foreclosed"
   ],
   "output": [
      "price"
   ]
}
```
