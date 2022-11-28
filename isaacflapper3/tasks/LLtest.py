import torch

class LLtest(object):

        def __init__(self):
                self.device = 'cuda'
                pass

        def make_trap_profile(self, c0, c1, s):
                y_n = torch.reshape((((torch.linspace(self.eps-1.0,1.0-self.eps,self.N, device=self.device))**2)**0.6)**0.5,[self.N,1]); # "station" locs
                idx = torch.tensor(torch.floor(self.N/2), dtype=torch.int32, device=self.device)
                y_n[0:idx,0] = -y_n[0:idx,0]
                y = torch.tensor(s*y_n, dtype=torch.float32,device=self.device)
                c = (y+s)*((c1-c0)/2.0/s)+c0

                return torch.unsqueeze(c, 0), torch.unsqueeze(y, 0)

        def setup(self):

                params = {"eps" : 1e-3,         # spacing from wing ends
                          "station_pts" : 20,     # nodes
                          "wings" : 2,           # number of wings
                          "envs" : 2,            # number of envs
                          "cla" : 6.5,             # cla
                          "rho" : 1000,             # density
                          "S1" : 0.075,              # wing 1 semispan
                          "S2" : 0.075,              # wing 2 semispan                  
                          "C1" : [0.1, 0.1],              # wing 1 chord
                          "C2" : [0.1, 0.1]}             # wing 2 chord

                self.eps = torch.tensor(params["eps"], device=self.device)
                self.N = torch.tensor(params["station_pts"], device=self.device)
                self.W = torch.tensor(params["wings"], device=self.device)
                self.M = torch.tensor(params["envs"], device=self.device)
                self.Cla = torch.tensor(params["cla"], device=self.device)
                self.S1 = torch.tensor(params["S1"], device=self.device)
                self.S2 = torch.tensor(params["S2"], device=self.device)
                self.C1 = torch.tensor(params["C1"], device=self.device)
                self.C2 = torch.tensor(params["C2"], device=self.device)

                self.s = torch.reshape(torch.tensor((params["S1"],params["S2"]), device=self.device), [2,1,1])

                C1, Y1 = self.make_trap_profile(params["C1"][0], params["C1"][1], self.s[0])
                C2, Y2 = self.make_trap_profile(params["C2"][0], params["C2"][1], self.s[1])

                self.C = torch.cat([C1, C2], dim=0)
                self.Y = torch.cat([Y1, Y2], dim=0) 

                self.theta = torch.acos(self.Y/self.s)
                self.vec1 = torch.sin(self.theta)*self.C*self.Cla/8.0/self.s

                self.n = torch.reshape(torch.linspace(1,self.N,self.N, dtype=torch.float32, device=self.device),[1,self.N])
                self.mat1 = (self.n*self.C*self.Cla/8.0/self.s + torch.sin(self.theta))*torch.sin(self.n*self.theta)
                self.mat2 = 4.0*self.s*torch.sin(self.n*self.theta)
                # Used in drag calculation 
                self.mat3 = torch.sin(self.n*self.theta)
                self.vec3 = torch.tensor(torch.reshape(torch.arange(1,self.N+1,device=self.device), (self.N,1))/torch.sin(self.theta), dtype=torch.float, device=self.device)

                self.force_scale = torch.squeeze(2*self.s/(self.N), dim=-1)

                print(self.mat1.shape)

def main():
        LLtest().setup()
        # pass

if __name__ == '__main__':
        main()