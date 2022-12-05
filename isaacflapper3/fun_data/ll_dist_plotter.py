import torch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import fftpack
import numpy as np

class Analysis(object):


	def __init__(self):


		self.path = "/home/afarah/Documents/UML/willis/rluff/isaacflapper3/fun_data"
		LiftDist_filename = "/LiftDist.pt"
		DOFstates_filename = "/DOFstates.pt"
		Rootstates_filename = "/Rootstates.pt"
		WingCOMs_filename = "/WingCOMs.pt"

		self.env = 0
		self.t_start = 0
		self.t_max = 999
		self.dt = 0.01
		self.num_stations = 20	
		self.force_scale = 0.01

		LiftDist = 1000*torch.load(self.path+LiftDist_filename)/self.force_scale
		DOFstates = torch.load(self.path+DOFstates_filename)
		Rootstates = torch.load(self.path+Rootstates_filename)
		WingCOMs = torch.load(self.path+WingCOMs_filename)

		self.LiftDistData = LiftDist[self.t_start:self.t_max,self.env,:,:].cpu().detach().numpy()
		self.DOFstatesData = DOFstates[self.t_start:self.t_max,int(self.env/2),:,:].cpu().detach().numpy()
		self.RootstatesData = Rootstates[self.t_start:self.t_max,int(self.env/2),:,:].cpu().detach().numpy()
		self.WingCOMsData = WingCOMs[self.t_start:self.t_max,int(self.env/2),:,:].cpu().detach().numpy()

		self.DOF_stroke_plane_pos = self.DOFstatesData[:,0,0]
		self.DOF_wing_1_pos = self.DOFstatesData[:,1,0]
		self.DOF_wing_2_pos = self.DOFstatesData[:,2,0]

		# plot root states on separate 3d plot
		self.Root_pos = self.RootstatesData[:,0,0:3]
		self.Root_linvel = self.RootstatesData[:,0,7:10]

		# plot wing coms on separate 3d plot
		self.Wing_1_COM = self.WingCOMsData[:,0,:] - self.Root_pos
		self.Wing_2_COM = self.WingCOMsData[:,1,:] - self.Root_pos

	def plot(self):
		fig1 = plt.figure()
		ax1 = plt.axes(projection='3d')	

		fig2 = plt.figure()
		ax2 = plt.axes(projection='3d')
		ax2.set_xlim3d(-10, 10)
		ax2.set_ylim3d(-10, 10)
		ax2.set_zlim3d(-10, 10)

		fig3 = plt.figure()
		ax3 = plt.axes(projection='3d')

		xs,ys = np.meshgrid(np.arange(1,self.LiftDistData.shape[0]), np.arange(0,self.num_stations,1))

		# plot the time evolution of the lift dists
		for t in range(int((np.amax(xs))/2)):
			t = 2*t-1
			LiftDistData_t = self.LiftDistData[t,...]
			xs_t = (xs[:,t] + self.t_start)*self.dt
			ys_t = ys[:,t]
			ax1.plot(xs_t,ys_t,LiftDistData_t.flatten())

		# plot the dof states
		x_dofs = np.arange(self.t_start,self.t_max,1)[0::]*self.dt
		y_dofs = np.zeros(self.t_max-self.t_start)
		# z_dofs = 100*self.DOF_wing_1_pos
		# z_dofs = 100*self.DOF_stroke_plane_pos

		ax1.plot(x_dofs, y_dofs, 10000*self.DOF_wing_2_pos, color='k')
		ax1.plot(x_dofs, y_dofs, 10000*self.DOF_stroke_plane_pos, color='r')
		ax1.set(xlabel='Time (s)', ylabel='Station', zlabel='Lift Dist [N/m]')
		ax1.set_title("Drag regime: Lift Distribution [N/m]")

		ax2.plot(self.Root_pos[:,0], self.Root_pos[:,1], self.Root_pos[:,2], color='k')
		ax2.set(xlabel='X [m]', ylabel='Y [m]', zlabel='Z [m]')
		ax2.set_title("Drag regime: Base Position [m]")

		ax3.plot(self.Wing_1_COM[:,0], self.Wing_1_COM[:,1], self.Wing_1_COM[:,2], color='k')
		ax3.plot(self.Wing_2_COM[:,0], self.Wing_2_COM[:,1], self.Wing_2_COM[:,2], color='k')
		ax3.set(xlabel='X [m]', ylabel='Y [m]', zlabel='Z [m]')
		ax3.set_title("Drag regime: Wing COM Position [m]")

		plt.show()

	def fft(self):
		x = np.arange(self.t_start,self.t_max,1)
		y = self.DOF_stroke_plane_pos
		print(x.shape, y.shape)

		yf = fftpack.fft(y, x.size)

		amp = np.abs(yf) # get amplitude spectrum 
		freq = np.linspace(0.0, 1.0/(2.0*(6/500)), x.size//2) # get freq axis
		# print(freq)

		# plot the amp spectrum
		print(self.env)
		print(f"Max frequency: {freq[np.argmax(amp)]}")
		print(f"Stroke plane amplitude: {np.max(np.abs(self.DOF_wing_2_pos)) - np.min(np.abs(self.DOF_wing_2_pos))}")
		print(f"Distance travelled: {((self.Root_pos[-1,0] - self.Root_pos[0,0])**2 + (self.Root_pos[-1,1] - self.Root_pos[0,1])**2 + (self.Root_pos[-1,2] - self.Root_pos[0,2])**2)**(1/2)}")

		plt.figure(figsize=(10,6))
		plt.plot(freq, (2/amp.size)*amp[0:amp.size//2])
		plt.show()
		pass 

	def strouhal_number(self):

		U = np.arange(0.0,0.6,0.1)
		A = np.array([0.28,0.19,0.40,0.46,0.42,0.51])
		f = np.array([0.35,0.48,0.24,0.24,0.24,0.36])

		St = 2*f*A*0.15/U

		f, ax = plt.subplots()
		ax.scatter(U,St)
		ax.grid()
		ax.set(xlabel="Base velocity [m/s]", ylabel="Strouhal number")
		ax.set_title("Strouhal number vs velocity")
		plt.show()

	def dist_from_flap(self):
		# distance travelled due to flap
		U = np.arange(0.0,0.6,0.1)
		elapsed_time = (self.t_max - self.t_start)*self.dt
		total_dist = np.array([0.15,0.7,1.39,2.09,2.79,3.5])
		dist_from_init_vel = U*elapsed_time
		dist_flap = total_dist - dist_from_init_vel

		f, ax = plt.subplots()
		ax.scatter(U,dist_flap)
		ax.grid()
		ax.set(xlabel="Base velocity [m/s]", ylabel="Distance due to flapping [m]")
		ax.set_title("Distance due to flapping vs velocity")
		plt.show()


if __name__ == '__main__':
	Analysis = Analysis()
	Analysis.plot()
	# Analysis.fft()
	# Analysis.strouhal_number()
	# Analysis.dist_from_flap()
