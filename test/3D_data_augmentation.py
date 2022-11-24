import numpy as np
from mayavi import mlab
import pickle
import os
from scipy.ndimage import distance_transform_edt as dte
from PIL import Image
import random
import math
import matplotlib.pyplot as plt
import time
import seaborn as sns

def read_pkl(filename) :
    file = open(filename, "rb")
    output = pickle.load(file)
    file.close()
    return output

def modify_branches(branches, bifs, st_pts) :
    offset = np.array([0,0,0])
    diff_offset = 0
    bifs = [np.array([bif[0],bif[1],0]) for bif in bifs]+[np.array([0,0,0])]
    new_branches = []

    bifs = st_pts[0:1]+bifs
    previous_pt = st_pts[0][:2]
    #for i in range(len(branches)) :
    for i in range(3) :
        
        diff_offset = bifs[i][2]
        print('diff_offset :',diff_offset)
        
        b_copy = np.zeros((len(branches[i]),3))
        b_copy[:,:2] = branches[i]

        # pick 1 angle first :
        angle = random.randrange(start=-70, stop=70)
        if i == 0 : angle = 10
        elif i == 1 : angle = 30
        elif i == 2 : angle = 45

        print('\nAngle :',angle,'\n')
        c = np.tan(math.radians(angle))

        b_copy[:,2] = np.linalg.norm(branches[i] - bifs[i][:2], axis=1) * c + diff_offset



        bifs[i:] = [np.array([x[0],x[1],np.sqrt(sum((x[:2]-previous_pt)**2))*c+diff_offset]) for x in bifs[i:]]

        previous_pt = bifs[i][:2]
        #offset[2] = bifs[i][2]
        #R_diff = np.linalg.norm(bifs[i][:2] - old_pt)
        #diff_offset += c * (R_diff) # scalar

        

        new_branches.append(np.round(b_copy).astype(int))
    
    return new_branches, bifs

def Rx(theta):
    return np.array([[ 1, 0              , 0              ],
                      [ 0, math.cos(theta),-math.sin(theta)],
                      [ 0, math.sin(theta), math.cos(theta)]])

def Ry(theta):
    return np.array([[ math.cos(theta) , 0, math.sin(theta)],
                      [ 0               , 1, 0              ],
                      [-math.sin(theta) , 0, math.cos(theta)]])

def Rz(theta):
    return np.array([[ math.cos(theta), -math.sin(theta) , 0 ],
                      [ math.sin(theta),  math.cos(theta) , 0 ],
                      [ 0              ,  0               , 1 ]])

def augment_data_3D(graphs, pts, bif_pts=None, nb_times=1) :

    new_graphs, new_pts, new_bifs = [], [], []
    a, b = -180, 180

    for i in range(len(graphs)) :
        
        for n in range(nb_times) :
            center = random.choice(pts[i])
        
            phi = random.randrange(start=a, stop=b)
            theta = random.randrange(start=a, stop=b)
            psi = random.randrange(start=a, stop=b)

            R = np.matmul(np.matmul(Rz(math.radians(psi)), Ry(math.radians(theta))), Rx(math.radians(phi)))

            # apply the rotation
            new_graph = np.matmul(R,(graphs[i]-center).T).T + center
            
            graph_mins = np.min(new_graph,axis=0)

            new_graphs.append(np.around(new_graph - graph_mins).astype(int))

            new_pts.append([np.around(np.matmul(R,(sub_pts-center).T).T + center - graph_mins).astype(int) for sub_pts in pts[i]])
            if bif_pts != None : new_bifs.append([np.around(np.matmul(R,(sub_pts-center).T).T + center - graph_mins).astype(int) for sub_pts in bif_pts[i]])
        
        graph_tmp = new_graphs[-1].copy()
        pts_tmp = new_pts[-1]
        g_min, g_max = np.minimum(np.min(graph_tmp, axis=0),np.array([0,0,0])), np.maximum(np.max(graph_tmp, axis=0),np.array([99,99,99]))
        _shape = g_max - g_min + np.array([1,1,1]) 
        origin = np.array(_shape)//2 + 0.5*(np.array(_shape)%2 - 1)        
        
        for sym_ax in range(graphs[i].shape[1]) : # along 3 axes
            gt_tmp = np.zeros(_shape)
            for g_pt in graph_tmp : gt_tmp[tuple(g_pt)] = 1
            gt_tmp = np.flip(gt_tmp, axis=sym_ax)
            new_graphs.append(np.argwhere(gt_tmp==1))
            new_pts.append([flip_point(pt, origin, axis=sym_ax) for pt in pts_tmp])
            if bif_pts != None : new_bifs.append([flip_point(b_pt, origin, axis=sym_ax) for b_pt in bif_tmp])

    if bif_pts != None :
        return new_graphs, new_pts, new_bifs
    else :
        return new_graphs, new_pts

def plot_axes(_shape) :
    L = 0.1
    xx = yy = zz = np.arange(0,_shape,0.5)
    xy = xz = yx = yz = zx = zy = np.zeros_like(xx)    
    mlab.plot3d(yx,yy+L,yz,line_width=0.1,tube_radius=0.3, color=(.3,.3,.3))
    mlab.plot3d(zx,zy+L,zz,line_width=0.1,tube_radius=0.3, color=(.3,.3,.3))
    mlab.plot3d(xx,xy+L,xz,line_width=0.1,tube_radius=0.3, color=(.3,.3,.3))

def to_3D(graphs, pts) :
    new_graphs, new_pts = [], []
    for i in range(len(graphs)):
        g_idx = np.argwhere(graphs[i]==1)
        # add 3rd dimension
        new_graphs.append(np.c_[g_idx, np.zeros(len(g_idx))])
        new_pts.append([np.array([sub_pt[0],sub_pt[1],0]) for sub_pt in pts[i]])
    return new_graphs, new_pts

def add_padding(img, pts, pad, bif_pts=None):
    if bif_pts :
        return np.pad(img,pad), [np.array(pt)+np.array([pad,pad]) for pt in pts], [np.array(pt)+np.array([pad,pad]) for pt in bif_pts]
    else :
        return np.pad(img,pad), [np.array(pt)+np.array([pad,pad]) for pt in pts]

def generate_gt(xline, yline, zline, pad) :
    gt = np.zeros((100,100,100))
    end_pts = []
    for p in [0,-1] :
        end_pts.append(np.array([int(xline[p]),int(yline[p]),int(zline[p])]))
    end_pts = [pt+np.array([pad,pad,pad]) for pt in end_pts]
    for i in range(len(zline)) :
        gt[int(xline[i]),int(yline[i]),int(zline[i])] = 1
    gt = np.pad(gt,pad)
    gt_idx = np.argwhere(gt == 1)
    return gt_idx, end_pts

def save_graph(saving_name, graph_idx, end_pts, bifs) :
    dictionary_data = {"graph_idx": graph_idx, "start_pts": end_pts, "bif_pts": bifs}
    pkl_file = open(saving_name, "wb")
    pickle.dump(dictionary_data, pkl_file)
    pkl_file.close()

def graph_to_3D_and_save() :

    # Retrieve the test graphs
    test_path = './graphs/100x'
    test_graphs_2, test_start_pts_2 = [], []

    for i in range(1,21) :
        pkl = read_pkl(os.path.join(test_path,'g'+str(i)+'.pkl'))
        G, pts = pkl['graph'], pkl['start_pts']
        test_graphs_2.append(G)
        test_start_pts_2.append(pts)
    
    graphs, pts = to_3D(test_graphs_2, test_start_pts_2)
    
    synth_gts, synth_pts = [], []
    pad = 0
    N = 400
    z = np.linspace(0, 99, N)
    x = (50*np.sin(z/15)+50)
    y = (50*np.sin(z/30)+50)
    gt1, end_pts = generate_gt(x, y, z, pad=pad)
    synth_gts.append(gt1); synth_pts.append(end_pts)

    z = np.linspace(0, 99, N)
    x = (50*np.cos(z/40)+49)
    y = (50*np.sin(z/30)+49)
    gt2, end_pts = generate_gt(y, z, x, pad=pad)
    synth_gts.append(gt2); synth_pts.append(end_pts)

    z = np.linspace(0, 99, N)
    x = (50*np.cos(z/18+1)+49)
    y = z**2; y = 99*y/y.max()
    gt3, end_pts = generate_gt(y, z, x, pad=pad)
    synth_gts.append(gt3); synth_pts.append(end_pts)

    z = np.linspace(0, 99, N)
    x = (50*np.cos(z/100)+49)
    y = (z/300)**3; y = 99*y/y.max()
    gt4, end_pts = generate_gt(x, y, z, pad=pad)
    synth_gts.append(gt4); synth_pts.append(end_pts)

    graphs = synth_gts + graphs
    pts = synth_pts + pts

    for i in range(len(graphs)) :
        saving_name = './graphs/3D_simple/g'+str(i)+'.pkl'
        save_graph(saving_name, graphs[i], pts[i], bifs=[])

def flip_point(pt, center, axis):
    flipped_pt = pt.copy()
    flipped_pt[axis] = 2*center[axis] - flipped_pt[axis]
    return flipped_pt

if __name__ == '__main__':

    # To generate and save the 3D graphs :
    # graph_to_3D_and_save()
    
    # To load them
    path_to_file = './graphs/3D_simple'
    graphs, pts = [], []
    for i in range(24) :
        output = read_pkl(os.path.join(path_to_file,'g'+str(i)+'.pkl'))
        graphs.append(output['graph_idx'])
        pts.append(output['start_pts'])

    # DON'T PAD BEFORE THAT TO NOT MESS WITH THE CENTER COORDINATES
    graphs, pts = augment_data_3D(graphs, pts, bif_pts=None, nb_times=5)

    if False :
        colors = [(0,1,1),(0,0.8,0.8),(0,0.6,0.6),(0,0.4,0.4),(0,0.2,0.2)]
        fig = mlab.figure(size=(1300,1000), bgcolor=(0.95,0.95,0.95))

        j = 5
        for i,ind in enumerate([0,1,2,3,4,5,6,7]) :
            
            GT, st_pts = graphs[ind+8*j], pts[ind+8*j]
            x, y, z = GT[:,0], GT[:,1], GT[:,2]
            mlab.points3d(x, y, z, scale_factor=1., color=colors[i%len(colors)], mode='cube')

            for pt in st_pts :
                mlab.points3d(pt[0], pt[1], pt[2], scale_factor=1.4, color=(0.7,0,0), mode='cube')

        plot_axes(_shape=100)
        mlab.show()

    
    colors = sns.color_palette()
    fig = mlab.figure(size=(1300,1000), bgcolor=(1,1,1))

    j = 7
    for i,ind in enumerate([0,1,2,3,4,5]) :
        
        GT, st_pts = graphs[ind+8*j], pts[ind+8*j]
        x, y, z = GT[:,0], GT[:,1], GT[:,2]
        mlab.points3d(x, y, z, scale_factor=1., color=colors[i%len(colors)], mode='cube')

    plot_axes(_shape=100)
    mlab.show()

    