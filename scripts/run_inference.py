


import numpy as np
import os, sys, argparse

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

from models.mlp import MLP
from models.cnn import CNN
from models.pointnet import PointNet
from utils.InferenceContainer import InferenceContainer as IC



if __name__ == '__main__':


	#Parse Arguments
	parser = argparse.ArgumentParser(description='Run inference on specified dataset')
	parser.add_argument('--dataset', help='dataset to be used (numpy format)', type=str, required=True)
	parser.add_argument('--labels', help='labels corresponding to dataset (numpy format)', type=str, required=True)
	parser.add_argument('--net', help='which type of network to run (mlp/cnn)', type=str, required=False, \
									choices=['mlp', 'cnn', 'pc'], default='mlp')
	parser.add_argument('--weights', help='folder containing network weights to use', type=str, required=True)

	args = parser.parse_args()

	d = np.load(args.dataset)
	l = np.load(args.labels)

	ic = IC(data=d, labels=l)

	pc = PointNet(n_points=ic.test.data.shape[1], weights_file=args.weights)

	acc = pc.inference(ic)

	print('inference accuracy: ' + str(acc))


