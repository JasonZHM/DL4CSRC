import torch 
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_

import numpy as np 
import matplotlib.pyplot as plt 

from flow import MongeAmpereFlow
import net 
import objectives

T = 0.1

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-cuda", type=int, default=-1, help="use GPU")
    parser.add_argument("-net", default='Simple_MLP', choices=['Simple_MLP', 'MLP'], 
                                                       help="network architecture")
    parser.add_argument("-target", default='Wave', 
                        choices=['Ring2D', 'Ring5', 'Wave', 'Gaussian', 'Mog2', 'electron'], help="target distribution")
    parser.add_argument("-numElectron", type=int, default=2)
    args = parser.parse_args()
    device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))

    xlimits=[-4, 4]
    ylimits=[-4, 4]
    numticks=31
    x = np.linspace(*xlimits, num=numticks, dtype=np.float32)
    y = np.linspace(*ylimits, num=numticks, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    xy = np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
    xy = torch.from_numpy(xy).contiguous().to(device)

    # Set up plotting code
    def plot_isocontours(ax, func, alpha=1.0):
        zs = np.exp(func(xy).cpu().detach().numpy())
        Z = zs.reshape(X.shape)
        plt.contour(X, Y, Z, alpha=alpha)
        ax.set_yticks([])
        ax.set_xticks([])
        plt.xlim(xlimits)
        plt.ylim(ylimits)

    if args.target=='electron':
        target = getattr(objectives, args.target)(args.numElectron)
    else: 
        target = getattr(objectives, args.target)()
    target.to(device)

    epsilon = 0.1 
    Nsteps = 10
    batch_size = 1024

    # Set up figure.
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)

    if args.target=='electron':
        net = getattr(net, args.net)(dim=2*args.numElectron, hidden_size=128, device=device)
    else: 
        net = getattr(net, args.net)(dim=2, hidden_size=32, device=device)
    model = MongeAmpereFlow(net, epsilon, Nsteps, device=device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)

    params = list(model.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)
    
    np_losses = []
    for e in range(1000):
        x, logp = model.sample(batch_size)
        loss = logp.mean() + target(x).mean()/T
        energy = target(x).mean()
        
        model.zero_grad()
        energy.backward()
        optimizer.step()

        # clip_grad_norm_(params, 1e1)
        
        with torch.no_grad():
            print (e, loss.item(), energy.item())
            np_losses.append([loss.item()])

            plt.cla()
            
            if args.target!='electron':  
                plot_isocontours(ax, target, alpha=0.5)
                plot_isocontours(ax, model.net) # Breiner potential 

            samples = x.cpu().detach().numpy()
            if args.target=='electron':
                for i in range(0, args.numElectron):
                    plt.plot(samples[:, i], samples[:,i+1],'o', alpha=0.8)
            else:
                plt.plot(samples[:, 0], samples[:,1],'o', alpha=0.8)

            plt.draw()
            plt.pause(0.01)

    np_losses = np.array(np_losses)
    fig = plt.figure(figsize=(8,8), facecolor='white')
    plt.ioff()
    plt.plot(np_losses[50:])
    plt.show()
