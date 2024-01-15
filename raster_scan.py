import numpy as np
import nidaqmx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from nidaq import Nidaq
#matplotlib.use('Qt5Agg')

def flip_every_second_row(A):
    B = A.copy()
    mat = B.copy()
    for i in range(len(B[:, 0])):
        if (i % 2) == 0:
            mat[i, :] = B[i, :]
        else:
            mat[i, :] = np.flip(B[i, :])
    return mat

def create_raster_array(c, w, volt_step, coordinates, guess):
    coordinates = np.sort(coordinates)
    guessAxes = np.setdiff1d([0, 1, 2, 3], coordinates)
    xcoords = np.arange(c[0] - w[0] / 2, c[0] + w[0] / 2 + volt_step, volt_step)
    ycoords = np.arange(c[1] - w[1] / 2, c[1] + w[1] / 2 + volt_step, volt_step)
    total_steps = len(xcoords)
    mat = np.zeros([4, total_steps])
    for i, V in enumerate(ycoords):
        if (i % 2) == 0:
            mat[coordinates[0], :] = xcoords
        else:
            mat[coordinates[0], :] = np.flip(xcoords)
            #mat[coordinates[0], :] = xcoords
        mat[coordinates[1], :] = np.ones([1, total_steps]) * V
        if i == 0:
            raster_positions = mat.copy()
        else:
            raster_positions = np.concatenate((raster_positions, mat), axis=1)

    raster_positions[guessAxes, :] = np.concatenate((np.ones([1, len(raster_positions[0, :])])*guess[0], np.ones([1, len(raster_positions[0, :])])*guess[1]),axis=0)
    return raster_positions

def single_raster_scan(dq, c, w, volt_step, coordinates, guess, simulate = False):
    guessAxes = np.setdiff1d([0, 1, 2, 3], coordinates)

    xcoords = np.arange(c[0] - w[0] / 2, c[0] + w[0] / 2 + volt_step, volt_step)
    ycoords = np.arange(c[1] - w[1] / 2, c[1] + w[1] / 2 + volt_step, volt_step)
    raster_positions = create_raster_array(c, w, volt_step, coordinates, guess)
    initial_volt = np.zeros(4)
    initial_volt[coordinates[0]] = c[0] - w[0]/2
    initial_volt[coordinates[1]] = c[1] - w[1] / 2
    initial_volt[guessAxes] = guess

    dq.set_volt(initial_volt)
    time.sleep(1)
    if simulate == False:
        raster_results = dq.daq_experiment(raster_positions, sr = 300, timeout = 5000)
    else:
        raster_results = dq.dist.pdf(np.transpose(raster_positions[0:3, :]))
    local_max = raster_positions[coordinates, np.argmax(raster_results)]
    max_val = np.max(raster_results)
    coordinates_dict = ["X", "Y", "Z", "\u03B8x", "\u03B8y", "\u03B8z"]
    # plot results
    X, Y = np.meshgrid(xcoords, ycoords)
    Zold = raster_results.reshape(len(ycoords), len(xcoords))
    Z = flip_every_second_row(Zold)
    #for i in range(len(ax_coords)):
     #   if (i % 2) == 0:
      #      Z[i, :] = Z[i, :]
      #  else:
       #     Z[i, :] = np.flip(Z[i, :])
    #Z[1::2, :] = np.flip(Z[1::2, :])
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    ax.set_xlabel(coordinates_dict[coordinates[0]] + " DAQ voltage [V]", fontsize=10, rotation=0)
    ax.set_ylabel(coordinates_dict[coordinates[1]] + " DAQ voltage [V]", fontsize=10, rotation=0)
    ax.set_zlabel("Photodetector voltage [V]", fontsize=10, rotation=0)

    plt.show()

    return raster_results, raster_positions, local_max, max_val, Z, X, Y


def scan_single_axis(dq, minV, maxV, voltStep, coordinate, guess):
    guessAxes = np.setdiff1d([0, 1, 2, 3], coordinate)
    ax_coords = np.arange(minV, maxV + voltStep, voltStep)
    totalSteps = int((maxV - minV) / voltStep + 1)
    raster_positions = np.zeros([4, totalSteps])
    raster_positions[guessAxes, :] = np.concatenate(
        (np.ones([1, len(raster_positions[0, :])]) * guess[0], np.ones([1, len(raster_positions[0, :])]) * guess[1], np.ones([1, len(raster_positions[0, :])]) * guess[2]),
        axis=0)
    raster_positions[coordinate] = ax_coords
    initial_volt = np.zeros([4, 1])

    initial_volt[guessAxes] = np.array([guess]).T

    dq.set_volt(initial_volt)
    time.sleep(1)
    raster_results = dq.daq_experiment(raster_positions)
    plt.plot(ax_coords, raster_results)
    plt.show()
    return raster_results

def raster4Dplot(dq, voltStep, minV, maxV, coordinates, guess):
    coordinates = np.asarray(list(itertools.combinations([0, 1, 2, 3], 2)))
    for i in range(6):

        raster_positions = create_raster_array(minV, maxV, voltStep, coordinates[i], guess)
        raster_results = daq_experiment(dq, sr, raster_positions)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax_coords = np.arange(minV, maxV + voltStep, voltStep)
        X, Y = np.meshgrid(ax_coords, ax_coords)
        Z, new_guess = raster_results.T.reshape(len(ax_coords), len(ax_coords))
        Z[1::2, :] = np.flip(Z[1::2, :])
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel(coordinates_dict[coordinates[i, 0]], fontsize=20, rotation=0)
        ax.set_ylabel(coordinates_dict[coordinates[i, 1]], fontsize=20, rotation=0)
        plt.show()