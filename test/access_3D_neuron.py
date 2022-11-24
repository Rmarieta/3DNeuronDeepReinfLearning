from turtle import color
from morphio import Morphology, Option
import os
import pickle
import seaborn as sns
from mayavi import mlab
import numpy as np
from scipy.ndimage import distance_transform_edt as dte
from scipy import interpolate
import math
import random
from scipy import ndimage
import psutil
from psutil._common import bytes2human
import random
import  moviepy.editor as mpy

# Run from inside ./raphaels_project, graphs will be saved in graphs/3D_neuromorpho

# Functions for data augmentation
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

def flip_point(pt, center, axis):
    flipped_pt = pt.copy()
    flipped_pt[axis] = 2*center[axis] - flipped_pt[axis]
    return flipped_pt

# Adds rotations and symmetries
def augment_data_3D(graphs, pts, bif_pts=None, nb_times=1, neuron_3D=False) :

    new_graphs, new_pts, new_bifs = [], [], []
    a, b = -180, 180

    for i in range(len(graphs)) :

        '''if neuron_3D : # add the original one            
            new_graphs.append(graphs[i]); new_pts.append(pts[i])
            if bif_pts != None : new_bifs.append(bif_pts[i])'''
        
        for n in range(nb_times) :
            center = random.choice(pts[i])

            if neuron_3D :
                phi = random.choice([-180, -90, 0, 90, 180])
                theta = random.choice([-180, -90, 0, 90, 180])
                psi = random.choice([-180, -90, 0, 90, 180])
            else :
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
        if bif_pts != None : bif_tmp = new_bifs[-1]
        g_min, g_max = np.minimum(np.min(graph_tmp, axis=0),np.array([0,0,0])), np.maximum(np.max(graph_tmp, axis=0),np.array([99,99,99]))
        _shape = g_max - g_min + np.array([1,1,1]) 
        origin = np.array(_shape)//2 + 0.5*(np.array(_shape)%2 - 1)        
        
        for sym_ax in range(graphs[i].shape[1]) : # along 3 axes
            gt_tmp = np.zeros(_shape)
            for g_pt in graph_tmp : gt_tmp[tuple(g_pt)] = 1
            gt_tmp = np.flip(gt_tmp, axis=sym_ax)
            new_gt_tmp = np.argwhere(gt_tmp==1)
            new_mins = np.min(new_gt_tmp, axis=0)
            new_graphs.append(new_gt_tmp - new_mins)
            new_pts.append([flip_point(pt, origin, axis=sym_ax)-new_mins for pt in pts_tmp])
            if bif_pts != None : new_bifs.append([flip_point(b_pt, origin, axis=sym_ax)-new_mins for b_pt in bif_tmp])

    if bif_pts != None :
        return new_graphs, new_pts, new_bifs
    else :
        return new_graphs, new_pts

def add_padding(img, pts, pad, bif_pts=None):
    if bif_pts :
        return np.pad(img,pad), [np.array(pt)+pad for pt in pts], [np.array(pt)+pad for pt in bif_pts]
    else :
        return np.pad(img,pad), [np.array(pt)+pad for pt in pts]

def img_from_idx(graphs, pts, dilation_rate=0, pad=10) :
    gts, padded_pts = [], []
    for i in range(len(graphs)) :
        _shape = np.max(graphs[i],axis=0) + 1
        GT = np.zeros(_shape)
        for pt in graphs[i] : GT[tuple(pt)] = 1
        GT, padded_pt = add_padding(GT, pts[i], pad=pad)
        gts.append(GT); padded_pts.append(padded_pt)
    if dilation_rate > 0 :
        imgs = []
        struct1 = ndimage.generate_binary_structure(3, 1)
        for img in gts :
            if random.random() < dilation_rate : # dilate according to the rate
                imgs.append(np.clip(dte(1-ndimage.binary_dilation(img, structure=struct1, iterations=2).astype(int)),a_min=0,a_max=15))
            else :
                imgs.append(np.clip(dte(1-img),a_min=0,a_max=15))
    else :
        imgs = [np.clip(dte(1-img),a_min=0,a_max=15) for img in gts]
    return gts, imgs, padded_pts

def read_pkl(filename) :
    file = open(filename, "rb")
    output = pickle.load(file)
    file.close()
    return output

def save_graph(saving_name, graph_idx, soma_pts, end_pts, bifs, pred_size) :
    dictionary_data = {"graph_idx": graph_idx, "soma_pts": soma_pts, "end_pts": end_pts, "bif_pts": bifs, 'pred_size': pred_size}
    pkl_file = open(saving_name, "wb")
    pickle.dump(dictionary_data, pkl_file)
    pkl_file.close()

# function that animates the mayavi figure by rotating it
def make_frame(t):
    """ Generates and returns the frame for time t. """
    scene.scene.camera.azimuth(2)
    #mlab.view(elevation= 180, azimuth=360*t/duration) # camera angle
    return mlab.screenshot(antialiased=True) # return a RGB image

if __name__ == "__main__" :

    col_pal = sns.color_palette(n_colors=10)

    graph_name = "./graphs/3D_neuromorpho/3D_neuron"
    neuron_name = "./test/neuron_3D_morphos/Layer-2-3-Ethanol-"

    plot = True
    read_from_pkl = True
    create_gif = True

    graphs, pts = [], []

    # 18 > 1 > 17 > 3 > 19
    for id_pal, neuron_idx in enumerate([18]) :#[17,18,19] : #range(1,28) :
   
        if plot : mlab.figure(size=(1700,1200), bgcolor=(1,1,1))
        
        if read_from_pkl :
            output = read_pkl(graph_name+str(neuron_idx)+'.pkl')
            gt_idx   = output["graph_idx"]
            soma_pts = output["soma_pts"]
            bif_pts  = output["bif_pts"]
            end_pts  = output["end_pts"]

        else :

            scaling_factor = 3.5 # factor to upscale the size of the neuron

            cell = Morphology(neuron_name+str(neuron_idx)+'.CNG.swc')

            N = 800

            gt_idx = np.empty((0,3), dtype=int)

            all_branches = []
            current_branch = np.empty((0,3))

            all_pts = np.empty((0,3))
            end_pts = np.empty((0,3))
            soma_pts = np.empty((0,3))
            for i in range(len(cell.sections)):
                _sec = cell.section(i).points * scaling_factor

                all_pts = np.append(all_pts, _sec, axis=0)
                end_pts = np.append(end_pts, _sec[-1:], axis=0)
                x, y, z = _sec[:,0], _sec[:,1], _sec[:,2]
                #mlab.points3d(x, y, z, color=col_pal[i%10], scale_factor=3, mode='cube')
                
                # interpolate the curve
                if len(x) <= 2 : k = 1
                elif len(x) == 3 : k = 2
                else : k = 3
                tck, u = interpolate.splprep([x,y,z], k=k, s=0)
                x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
                u_fine = np.linspace(0,1,N)
                x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
                #mlab.plot3d(x_fine, y_fine, z_fine, color=col_pal[i%10], tube_radius=0.1)
                # discretize the curve
                x_int, y_int, z_int = np.around(x_fine).astype(int), np.around(y_fine).astype(int), np.around(z_fine).astype(int)
                
                #mlab.points3d(x_int, y_int, z_int, color=col_pal[2], scale_factor=1, mode='cube')

                branch_idx = np.concatenate([x_int[:,None],y_int[:,None],z_int[:,None]], axis=1)
                # remove duplicates
                branch_idx = np.unique(branch_idx, axis=0)
                gt_idx = np.append(gt_idx, branch_idx, axis=0)
                
                # to separately save the branches and all their children branches
                if i == 0 : current_branch = np.append(current_branch, branch_idx, axis=0)
                elif cell.section(i).is_root :
                    all_branches.append(current_branch)
                    current_branch = branch_idx
                else : 
                    current_branch = np.append(current_branch, branch_idx, axis=0)
                    if i == (len(cell.sections) - 1) : all_branches.append(current_branch)

            # First points of root_sections are always starting points on the soma side
            for root_sec in cell.root_sections :
                _sec2 = root_sec.points * scaling_factor
                soma_pts = np.append(soma_pts, _sec2[:1,:], axis=0)
            
            # Retrieve bifurcation points
            unq, count = np.unique(all_pts, axis=0, return_counts=True)
            bif_pts = unq[count>1]

            # Retrieve end points which can then be used as starting points
            end_pts = np.append(end_pts, bif_pts, axis=0)
            unq_2, count_2 = np.unique(end_pts, axis=0, return_counts=True)
            end_pts = unq_2[count_2==1]

            soma_pts = np.around(soma_pts).astype(int)
            bif_pts = np.around(bif_pts).astype(int)
            end_pts = np.around(end_pts).astype(int)

            p_min, p_max = np.min(gt_idx, axis=0), np.max(gt_idx, axis=0)

            # retrieve the various points branch-wise
            bif_set = set([tuple(x) for x in bif_pts])
            end_set = set([tuple(x) for x in end_pts])
            soma_set = set([tuple(x) for x in soma_pts])

            branch_name = './graphs/3D_neuromorpho_branches/neuron_'+str(neuron_idx)+'_'
            save = False
            if save :
                for l in range(len(all_branches)) :
                    b = all_branches[l].astype(int)
                    b = np.unique(b, axis=0)
                    b_set = set([tuple(x) for x in b])
                    mins, maxs = np.min(b, axis=0), np.max(b, axis=0)
                    bifs_branch = np.array([x for x in bif_set & b_set])
                    end_branch = np.array([x for x in end_set & b_set])
                    soma_branch = np.array([x for x in soma_set & b_set])

                    # shift all the point coordinates so that is starts at (0,0,0)
                    b -= mins     
                    if len(bifs_branch > 0) : bifs_branch -= mins    
                    if len(end_branch > 0)  : end_branch  -= mins
                    if len(soma_branch > 0) : soma_branch -= mins 

                    #if len(soma_branch > 0) : mlab.points3d(soma_branch[:,0], soma_branch[:,1], soma_branch[:,2], color=col_pal[l%10], scale_factor=5, mode='cube')
                    #if len(bifs_branch > 0) : mlab.points3d(bifs_branch[:,0], bifs_branch[:,1], bifs_branch[:,2], color=col_pal[l%10], scale_factor=5, mode='cube')
                    #if len(end_branch > 0) : mlab.points3d(end_branch[:,0], end_branch[:,1], end_branch[:,2], color=col_pal[l%10], scale_factor=5, mode='cube')
                    #mlab.points3d(b[:,0], b[:,1], b[:,2], color=col_pal[l%10], scale_factor=1, mode='cube')

                    _size = round(9.31e-7*np.zeros(maxs-mins).nbytes) # in MB
                    #print('Predicted size :',_size,'MB (neuron idx :',neuron_idx,')\n')

                    # to save as pickle files
                    save_graph(saving_name=branch_name+'branch_'+str(l)+'.pkl', graph_idx=b, soma_pts=soma_branch, end_pts=end_branch, bifs=bifs_branch, pred_size=_size)

            # shift all the point coordinates so that it starts at (0,0,0)
            gt_idx   -= p_min    
            soma_pts -= p_min    
            bif_pts  -= p_min    
            end_pts  -= p_min
            
            # save to pickle file
            saving_name = './graphs/3D_neuromorpho/3D_neuron'
            saving_name += str(neuron_idx)+'.pkl'
            full_size = round(9.31e-7*np.zeros(p_max-p_min).nbytes)
            #save_graph(saving_name=saving_name, graph_idx=gt_idx, soma_pts=soma_pts, end_pts=end_pts, bifs=bif_pts, pred_size=full_size)

        print(neuron_idx,'size :',np.max(gt_idx,axis=0)-np.min(gt_idx,axis=0),', in bytes :',bytes2human(np.zeros(np.max(gt_idx,axis=0)-np.min(gt_idx,axis=0)).nbytes))
        if plot :
            #mlab.points3d(soma_pts[:,0], soma_pts[:,1], soma_pts[:,2], color=(1,0.3,0.3), scale_factor=2, mode='cube')
            #mlab.points3d(bif_pts[:,0], bif_pts[:,1], bif_pts[:,2], color=(0.3,1,0.3), scale_factor=2, mode='cube')
            #mlab.points3d(end_pts[:,0], end_pts[:,1], end_pts[:,2], color=(1,0.9,0.3), scale_factor=2, mode='cube')
            mlab.points3d(gt_idx[:,0], gt_idx[:,1], gt_idx[:,2], color=(0.75,0,0), scale_factor=2, mode='cube')

        
        # to display the agent reconstruction around the ground truth
        if plot :
            output = read_pkl('./test/dump_scripts/neuron_'+str(neuron_idx)+'.pkl')
            pred_idx   = output["graph_idx"]
            pred_idx -= np.min(pred_idx,axis=0)
            pred_idx = np.unique(pred_idx,axis=0)
            print('Mins gt :',np.min(gt_idx,axis=0))
            print('Mins pred :',np.min(pred_idx,axis=0))
            mlab.points3d(pred_idx[:,0], pred_idx[:,1], pred_idx[:,2], color=(0.1,0.9,0.3), scale_factor=10, opacity=0.025, mode='cube')
        
        if plot : 
            
            if create_gif :
                scene = mlab.gcf()
                # change mayavi viewing point
                if neuron_idx == 18 :
                    scene.scene.camera.position = [367.64957423737053, 437.8400650715584, 1785.8987743936905]
                    scene.scene.camera.focal_point = [321.4530048735087, 401.450540444554, 157.0889240817067]
                    scene.scene.camera.view_angle = 30.0
                    scene.scene.camera.view_up = [-0.7027923059621458, -0.7104934253166569, 0.035805966852105464]
                    scene.scene.camera.compute_view_plane_normal()
                elif neuron_idx == 1 :
                    scene.scene.camera.position = [-287.2089958055607, -144.53888404934256, -1442.0222182767138]
                    scene.scene.camera.focal_point = [208.45556572850433, 408.9280163199513, 147.3252659247743]
                    scene.scene.camera.view_angle = 30.0
                    scene.scene.camera.view_up = [0.10310830377745864, -0.9488923373819135, 0.2982817623489983]
                    scene.scene.camera.compute_view_plane_normal()
                elif neuron_idx == 17 :
                    scene.scene.camera.position = [-492.4894491509599, 575.8916121867255, 1964.9706159134955]
                    scene.scene.camera.focal_point = [298.7802381039053, 450.27433504187843, 154.27245632622308]
                    scene.scene.camera.view_angle = 30.0
                    scene.scene.camera.view_up = [-0.15231220483204902, 0.9791394266703811, -0.13448782621039437]
                    scene.scene.camera.compute_view_plane_normal()
                else :
                    pass
                scene.scene.camera.clipping_range = [1030, 4000]
                duration = 9 # duration of the animation in seconds (it will loop)
                animation = mpy.VideoClip(make_frame, duration=duration)
                animation.write_gif("neuron_"+str(neuron_idx)+".gif", fps=20)

            
            mlab.show()

        graphs.append(gt_idx)
        pts.append(end_pts)


    augment_data = False
    if augment_data : 

        idx = [0,1]
        test_graphs = [graphs[i] for i in idx]
        test_start_pts = [pts[i] for i in idx]
    
        print(f'From {len(test_graphs)} graphs...')
        test_graphs, st_pts = augment_data_3D(test_graphs, test_start_pts, bif_pts=None, nb_times=1, neuron_3D=True)
        print(f'... to {len(test_graphs)} graphs')

        for i in range(len(test_graphs)) :

            mlab.figure(size=(1700,1200), bgcolor=(0.95,0.95,0.95))
            
            gt_idx = test_graphs[i]
            end_pts = st_pts[i]
            for pt in end_pts :
                mlab.points3d(pt[0], pt[1], pt[2], color=(1,0.9,0.3), scale_factor=2, mode='cube')
            mlab.points3d(gt_idx[:,0], gt_idx[:,1], gt_idx[:,2], color=col_pal[0], scale_factor=1, mode='cube')

        mlab.show()


