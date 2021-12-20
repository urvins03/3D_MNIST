import h5py
import os
import sys
import plotly
import pickle
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.offline import iplot

with h5py.File("E:/3d_cnn_mnist/3d_mnist_data/full_dataset_vectors.h5", 'r') as hf:
    X_train = hf["X_train"][:]
    y_train = hf["y_train"][:]
    X_test = hf["X_test"][:]
    y_test = hf["y_test"][:]

print ("x_train shape: ", X_train.shape)
print ("y_train shape: ", y_train.shape)

print ("x_test shape:  ", X_test.shape)
print ("y_test shape:  ", y_test.shape)

with h5py.File("E:/3d_cnn_mnist/3d_mnist_data/train_point_clouds.h5", "r") as points_dataset:
    digits = []
    for i in range(10):
        digit = (points_dataset[str(i)]["img"][:],
                 points_dataset[str(i)]["points"][:],
                 points_dataset[str(i)].attrs["label"])
        digits.append(digit)

x_c = [r[0] for r in digits[0][1]]
y_c = [r[1] for r in digits[0][1]]
z_c = [r[2] for r in digits[0][1]]
trace1 = go.Scatter3d(x=x_c, y=y_c, z=z_c, mode='markers',
                      marker=dict(size=12, color=z_c, colorscale='Viridis', opacity=0.7))

data = [trace1]
layout = go.Layout(height=500, width=600, title="Digit: " + str(digits[0][2]) + " in 3D space")
fig = go.Figure(data=data, layout=layout)
iplot(fig)

"""
save train and test data using pickle
"""
#
# with open('X_train.pickle', 'wb') as output:
#     pickle.dump(X_train, output)

