
import sys
import pptk
import pandas as pd
import numpy as np


def visualize_key(pc, src, tgt):
    base_body_color = np.array([[1., 1., 1., 1.]])
    src_color = np.array([[1., 15/255., 0., 1.]])
    tgt_color = np.array([[0., 1., 15/255., 1.]])

    color = np.zeros((pc.shape[0], 4))
    color[tgt] = tgt_color
    color[src] = src_color
    color[~(src | tgt)] = base_body_color

    v = pptk.viewer(pc, color)
    v.set(point_size=0.005)
    v.set(phi=6.89285231)
    v.set(r=2.82503915)
    v.set(theta=0.91104609)
    v.set(lookat=[0.00296851, 0.01331535, -0.47486299])
    return v

df = pd.read_csv(sys.argv[1])
pc = df[['x', 'y', 'z']].to_numpy()
src = df[['src']].to_numpy().squeeze().astype('bool')
tgt = df[['tgt']].to_numpy().squeeze().astype('bool')

visualize_key(pc, src, tgt)
