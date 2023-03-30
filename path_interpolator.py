import ab_pure_pursuit.util as util
import speed_control.util as sc_util

import speed_control.path

import numpy as np
import scipy.interpolate as interpolate
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def gen_cubicspline(via_points: np.array):
    expanded_points = np.zeros(shape=(via_points.shape[0], 3))

    # iterate over all points along the path
    for i in range(via_points.shape[0]):
        expanded_points[i][0] = i
        expanded_points[i][1] = via_points[i][0]
        expanded_points[i][2] = via_points[i][1]

    interpolation = interpolate.CubicSpline(x=expanded_points[:,0], y=expanded_points[:,1:])
    return interpolation

def plot_interpolation(interpolation: interpolate.CubicSpline, via_points: np.array, fineness: float, ax: plt.Axes):
    interpolation_plot = np.zeros(shape=(via_points.shape[0] * int(fineness), 2))
    for i in range(interpolation_plot.shape[0]):
        point = interpolation(i / fineness)

        interpolation_plot[i][0] = point[0]
        interpolation_plot[i][1] = point[1]

    # plot the points in a scatter plot
    ax.scatter(interpolation_plot[:,0], interpolation_plot[:,1], s=2)
    return interpolation_plot

def plot_interpolation_derivative(interpolation, via_points, fineness, ax):
    derivative_plot = np.zeros(shape=(via_points.shape[0] * int(fineness), 2))

    derivative_plot[0] = via_points[0]
    for i in range(1, derivative_plot.shape[0]):
        gradient = interpolation((i - 1) / fineness)
        approx_point = derivative_plot[i - 1] + gradient

        derivative_plot[i] = approx_point

    # plot the points in a scatter plot
    ax.scatter(derivative_plot[:,0], derivative_plot[:,1], s=2)
    return derivative_plot

def gen_bspline(via_points: np.array):
    return interpolate.splprep([via_points[:,0], via_points[:,1]], s=0.01, k=5)

def compute_curvature(first_derivative, second_derivative):
    numerator = np.linalg.norm(np.cross(first_derivative, second_derivative))
    denominator = np.linalg.norm(first_derivative)**3

    return numerator / denominator

def main():

    # load a set of points from file
    file = '/home/anthony/catkin_ws/src/ab_pure_pursuit/data/mod_path.csv'
    via_points = util.import_waypoints(file=file)

    bspline = gen_bspline(via_points)
    cubicspline = gen_cubicspline(via_points)

    # plot the points in a scatter plot
    plt.plot(via_points[:,0], via_points[:,1], 'ro')

    # plotting b-spline
    new_points = interpolate.splev(bspline[1], bspline[0])
    # plt.plot(new_points[0], new_points[1], 'r-')

    # plotting b-spline with derivatives
    derivative_points = interpolate.splev(bspline[1], bspline[0], der=1)
    derivative_points = np.array([derivative_points[0], derivative_points[1]])
    derivative_points = derivative_points.T
    
    scale_factor = cubicspline.derivative()(0)[0] / derivative_points[0,0]
    derivative_points *= scale_factor

    # print(derivative_points[0])
    # print(cubicspline.derivative()(0))
    # print(scale_factor)

    second_derivative_points = interpolate.splev(bspline[1], bspline[0], der=2)
    second_derivative_points = np.array([second_derivative_points[0], second_derivative_points[1]])
    second_derivative_points = second_derivative_points.T

    scale_factor = cubicspline.derivative().derivative()(0)[0] / second_derivative_points[0,0]
    second_derivative_points *= scale_factor

    print(derivative_points[0])
    print(second_derivative_points[0])

    integrated_points = np.zeros(via_points.shape)
    integrated_points[0] = via_points[0]
    for i in range(1, integrated_points.shape[0]):
        integration = integrated_points[i - 1] + (derivative_points[i - 1] + derivative_points[i]) * 0.5
        integrated_points[i] = integration

    plt.plot(integrated_points[:,0], integrated_points[:,1], 'r-')
    plt.show()

    path = speed_control.path.Path(filepath=file)

    ax = plt.axes(projection='3d')

    speeds = np.empty(via_points.shape[0], dtype=np.float)
    curvatures = np.empty(via_points.shape[0], dtype=np.float)

    modified_curvatures = np.empty(via_points.shape[0], dtype=np.float)

    for i in range(via_points.shape[0]):
        curvature = sc_util.compute_curvature(path, via_points[i])
        curvatures[i] = curvature
        modified_curvatures[i] = compute_curvature(derivative_points[i], second_derivative_points[i])
        speeds[i] = sc_util.compute_speed(max_accel=1.0, curvature=curvature)
        speeds[i] = max(min(1.0, speeds[i]), 0.5)

        # print(f'speed={speeds[i]}\ninterpolation_plot={str(interpolation_plot[i])}')

    # Data for three-dimensional scattered points
    ax.plot3D(via_points[:,0], via_points[:,1], curvatures)
    ax.plot3D(via_points[:,0], via_points[:,1], modified_curvatures)
    # ax.plot3D(via_points[:,0], via_points[:,1], speeds)
    plt.show()

if __name__ == '__main__':
    main()