import torch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

def main():
	fig = plt.figure()
	ax = plt.axes(projection='3d')

	path = "/home/afarah/Documents/UML/willis/rluff/isaacflapper3/fun_data"
	filename = "/ll_dist_data.pt"

	env = 0
	LL_dist = torch.load(path+filename)
	data = LL_dist[1:,env,:,:].cpu().detach().numpy()

	xs,ys = np.meshgrid(np.arange(1,LL_dist.shape[0]), np.arange(0,20,1))

	ax.scatter3D(xs,ys,data.flatten())

	plt.show()

if __name__ == '__main__':
	main()