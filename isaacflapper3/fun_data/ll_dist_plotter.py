import torch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

def main():
	fig1 = plt.figure()
	ax1 = plt.axes(projection='3d')	

	fig2 = plt.figure()
	ax2 = plt.axes(projection='3d')

	fig3 = plt.figure()
	ax3 = plt.axes(projection='3d')

	path = "/home/afarah/Documents/UML/willis/rluff/isaacflapper3/fun_data"
	LiftDist_filename = "/LiftDist.pt"
	DOFstates_filename = "/DOFstates.pt"
	Rootstates_filename = "/Rootstates.pt"
	WingCOMs_filename = "/WingCOMs.pt"

	env = 0
	t_max = 1000
	dt = 0.001

	LiftDist = torch.load(path+LiftDist_filename)
	DOFstates = torch.load(path+DOFstates_filename)
	Rootstates = torch.load(path+Rootstates_filename)
	WingCOMs = torch.load(path+WingCOMs_filename)

	LiftDistData = LiftDist[1:,env,:,:].cpu().detach().numpy()
	DOFstatesData = DOFstates[1:,env,:,:].cpu().detach().numpy()
	RootstatesData = Rootstates[1:,env,:,:].cpu().detach().numpy()
	WingCOMsData = WingCOMs[1:,env,:,:].cpu().detach().numpy()

	xs,ys = np.meshgrid(np.arange(1,LiftDistData.shape[0]), np.arange(0,20,1))

	# plot the time evolution of the lift dists
	for t in range(np.amax(xs)):
		LiftDistData_t = LiftDistData[t,...]
		xs_t = xs[:,t]
		ys_t = ys[:,t]
		ax1.plot(xs_t,ys_t,LiftDistData_t.flatten())

	# plot the dof states
	DOF_stroke_plane_pos = DOFstatesData[:,0,0]
	DOF_wing_1_pos = DOFstatesData[:,1,0]
	DOF_wing_2_pos = DOFstatesData[:,2,0]

	x_dofs = np.arange(1,t_max,1)
	y_dofs = np.zeros(t_max-1)
	z_dofs = 10*DOF_wing_1_pos

	ax1.plot(x_dofs, y_dofs, z_dofs, color='k')
	ax1.set(xlabel='Time (s)', ylabel='Station', zlabel='Lift Dist [N/m]')

	# plot root states on separate 3d plot
	Root_pos = RootstatesData[:,0,0:3]
	Root_linvel = RootstatesData[:,0,7:10]

	x_pos = Root_pos[:,0]
	y_pos = Root_pos[:,1]
	z_pos = Root_pos[:,2]

	ax2.plot(Root_pos[:,0], Root_pos[:,1], Root_pos[:,2], color='k')
	ax2.set(xlabel='X [m]', ylabel='Y [m]', zlabel='Z [m]')
	
	# plot wing coms on separate 3d plot
	Wing_1_COM = WingCOMsData[:,0,:]
	Wing_2_COM = WingCOMsData[:,1,:]

	ax3.plot(Wing_1_COM[:,0], Wing_1_COM[:,1], Wing_1_COM[:,2], color='k')
	ax3.plot(Wing_2_COM[:,0], Wing_2_COM[:,1], Wing_2_COM[:,2], color='k')
	ax3.set(xlabel='X [m]', ylabel='Y [m]', zlabel='Z [m]')
	
	plt.show()

if __name__ == '__main__':
	main()