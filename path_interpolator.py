import ab_pure_pursuit.util as util
import speed_control.util as sc_util

import speed_control.path

import numpy as np
import scipy.interpolate as interpolate
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def main():

    # load a set of points from file
    file = '/home/anthony/catkin_ws/src/ab_pure_pursuit/data/track_nitish.csv'
    via_points = util.import_waypoints(file=file)

    figure, axs = plt.subplots(nrows=1, ncols=3)

    # plot the points in a scatter plot
    axs[0].scatter(via_points[:,0], via_points[:,1], s=5)

    expanded_points = np.zeros(shape=(via_points.shape[0], 3))

    # iterate over all points along the path
    for i in range(via_points.shape[0]):
        expanded_points[i][0] = i
        expanded_points[i][1] = via_points[i][0]
        expanded_points[i][2] = via_points[i][1]

    interpolation = interpolate.CubicSpline(x=expanded_points[:,0], y=expanded_points[:,1:])

    fineness = 1.0

    interpolation_plot = np.zeros(shape=(via_points.shape[0] * int(fineness), 2))
    for i in range(interpolation_plot.shape[0]):
        point = interpolation(i / fineness)

        interpolation_plot[i][0] = point[0]
        interpolation_plot[i][1] = point[1]

    # plot the points in a scatter plot
    axs[1].scatter(interpolation_plot[:,0], interpolation_plot[:,1], s=2)

    derivative_interpolation = interpolation.derivative()
    derivate_plot = np.zeros(shape=(via_points.shape[0] * int(fineness), 2))

    derivate_plot[0] = via_points[0]
    for i in range(1, derivate_plot.shape[0]):
        gradient = derivative_interpolation((i - 1) / fineness)
        approx_point = derivate_plot[i - 1] + gradient

        derivate_plot[i] = approx_point

    # plot the points in a scatter plot
    axs[2].scatter(derivate_plot[:,0], derivate_plot[:,1], s=2)

    plt.show()
    print(interpolation.derivative()(1))
    print(interpolation.derivative().derivative()(1))

    path = speed_control.path.Path(filepath=file)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    speeds = np.empty(interpolation_plot.shape[0], dtype=np.float)
    curvatures = np.empty(interpolation_plot.shape[0], dtype=np.float)
    for i in range(interpolation_plot.shape[0]):
        curvature = sc_util.compute_curvature(path, interpolation_plot[i])
        curvatures[i] = curvature
        speeds[i] = sc_util.compute_speed(max_accel=1.0, curvature=curvature)
        speeds[i] = max(min(1.0, speeds[i]), 0.5)

        print(f'speed={speeds[i]}\ninterpolation_plot={str(interpolation_plot[i])}')

    # Data for three-dimensional scattered points
    ax.plot3D(interpolation_plot[:,0], interpolation_plot[:,1], curvatures)
    plt.show()

if __name__ == '__main__':
    main()