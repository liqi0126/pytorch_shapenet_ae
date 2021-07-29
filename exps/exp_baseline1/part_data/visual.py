import os
import pptk
import pandas as pd
import numpy as np

if __name__ == '__main__':
    for file in os.listdir("."):
        df = pd.read_csv(file)
        pc = df[['x', 'y', 'z']].to_numpy()
        key = df[['key']].to_numpy().squeeze().astype('bool')

        base_body_color = np.array([[1., 1., 1., 1.]])
        key_color = np.array([[135 / 255., 205 / 255., 92 / 255., 1.]])

        color = np.zeros((pc.shape[0], 4))
        color[key] = key_color
        color[~key] = base_body_color

        v = pptk.viewer(pc, color)
        v.set(point_size=0.005)
        v.set(phi=6.89285231)
        v.set(r=2.82503915)
        v.set(theta=0.91104609)
        v.set(lookat=[0.00296851, 0.01331535, -0.47486299])
        import ipdb; ipdb.set_trace()
        v.close()