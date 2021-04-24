import torch
import numpy as np
from torch.nn.functional import pdist, relu
from .template import Target

class electron(Target):

    def __init__(self, numElectron, wallSlope=1e2):
        super(electron, self).__init__(2*numElectron,'electron') # num of variables, name
        self.numElectron = numElectron
        self.wallSlope = wallSlope

    def energy(self, x):
        interaction = torch.sum(
            1/torch.stack(list(map(pdist, torch.split(x.reshape(-1, 2), self.numElectron))), dim=0).reshape(len(x), -1), 
            axis=1).reshape(-1, 1)
        # wall = torch.sum(relu((x+4)*(x-4)), dim=1) # square
        # wall = self.wallSlope * relu(torch.sqrt(torch.sum(x**2, dim=1))-5) # circle
        wall = torch.sum(x**2, dim=1)/100
        return interaction + wall