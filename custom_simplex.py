# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:59:09 2023

@author: s176369
"""
import numpy as np
import nidaq
import time
import matplotlib.pyplot as plt
def rearange(P1, P1val, P2, P2val, P3, P3val, P4 = None, P4val = None):
    if P4 is None:
        if P3val < P1val:  # If Rval is larger than Hval then switch them.
            P1, P3 = P3, P1
            P1val, P3val = P3val, P1val
        if P2val < P1val:
            P1, P2 = P2, P1
            P1val, P2val = P2val, P1val
        if P3val < P2val:
            P3, P2 = P2, P3
            P3val, P2val = P2val, P3val
    else:
        if P4val < P3val:
            P4, P3 = P3, P4
            P4val, P3val = P3val, P4val
        if P4val < P2val:
            P4, P2 = P2, P4
            P4val, P2val = P2val, P4val
        if P4val < P1val:
            P4, P1 = P1, P4
            P4val, P1val = P1val, P4val
        if P3val < P2val:
            P3, P2 = P2, P3
            P3val, P2val = P2val, P3val
        if P3val < P1val:
            P3, P1 = P1, P3
            P3val, P1val = P1val, P3val
        if P2val < P1val:
            P2, P1 = P1, P2
            P2val, P1val = P1val, P2val
    return P1, P1val, P2, P2val, P3, P3val, P4, P4val

def reflect_triangle(P1, P2, P3, P4 = None): #Make sure that the points are put in right order. It reflects R1 in the plane of the triangle spanned by the other points
    if P4 is None:
        return P3 + (P2 - P1)
    else:
        return P1 - 2 * (P1 - find_center_of_triangle(P2, P3, P4))

def plot_triangle_2d(L, Lval, M, Mval, H, Hval, R, Rval, E = None, Eval = None, Ci = None
                     , Cival = None, Cir = None, Cirval = None, Ce = None, Ceval = None,
                     Cer = None, Cerval = None):

    plt.scatter([L[0], M[0], H[0]], [L[1], M[1], H[1]])
    plt.scatter([R[0]], [R[1]])
    plt.text(L[0], L[1], "L = " + str(Lval))
    plt.text(M[0], M[1], "M = " + str(Mval))
    plt.text(H[0], H[1], "H = " + str(Hval))
    plt.text(R[0], R[1], "R = " + str(Rval))

    if E is not None:
        plt.scatter(E[0], E[1], color = "r")
        plt.text(E[0], E[1], "E = " + str(Eval))
    if Ci is not None:
        plt.scatter(Ci[0], Ci[1], color = "r")
        plt.text(Ci[0], Ci[1], "Ci = " + str(Cival))
    if Cir is not None:
        plt.scatter(Cir[0], Cir[1], color = "r")
        plt.text(Cir[0], Cir[1], "Cir = " + str(Cirval))
    if Ce is not None:
        plt.scatter(Ce[0], Ce[1], color = "r")
        plt.text(Ce[0], Ce[1], "Ce = " + str(Ceval))
    if Cer is not None:
        plt.scatter(Cer[0], Cer[1], color = "r")
        plt.text(Cer[0], Cer[1], "Cer = " + str(Cerval))
    plt.show()
def find_center_of_triangle(P1, P2, P3):
    O = 0.5 * (P3 - P2)
    Y = O + P2
    return (Y - P1) * 2/3 + P1

def plot_triangle_3d(P1, P1val, P2, P2val, P3, P3val, P4, P4val, R, Rval, E = None, Eval = None, Ci = None
                     , Cival = None, Cir = None, Cirval = None, Ce = None, Ceval = None,
                     Cer = None, Cerval = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([P1[0], P2[0], P3[0], P4[0]], [P1[1], P2[1], P3[1], P4[1]], [P1[2], P2[2], P3[2], P4[2]], c='b', marker='o')
    ax.scatter(R[0], R[1], R[2], color="k")
    ax.text(P1[0], P1[1], P1[2], "P1 = " + str(P1val))
    ax.text(P2[0], P2[1], P2[2], "P2 = " + str(P2val))
    ax.text(P3[0], P3[1], P3[2], "P3 = " + str(P3val))
    ax.text(P4[0], P4[1], P4[2], "P4 = " + str(P4val))
    ax.text(R[0], R[1], R[2], "R = " + str(Rval))
    ax.set_xlabel("X DAQ volt")
    ax.set_ylabel('Y DAQ volt')
    ax.set_zlabel('Z DAQ volt')
    if E is not None:
        ax.scatter(E[0], E[1], E[2], color="r")
        ax.text(E[0], E[1], E[2], "E = " + str(Eval))
    if Ci is not None:
        ax.scatter(Ci[0], Ci[1], Ci[2], color="r")
        ax.text(Ci[0], Ci[1], Ci[2], "Ci = " + str(Cival))
    if Cir is not None:
        ax.scatter(Cir[0], Cir[1], Cir[2], color="r")
        ax.text(Cir[0], Cir[1], Cir[2], "Cir = " + str(Cirval))
    if Ce is not None:
        ax.scatter(Ce[0], Ce[1], Ce[2], color="r")
        ax.text(Ce[0], Ce[1], Ce[2], "Ce = " + str(Ceval))
    if Cer is not None:
        ax.scatter(Cer[0], Cer[1], Cer[2], color="r")
        ax.text(Cer[0], Cer[1], Cer[2], "Cer = " + str(Cerval))
    plt.show()
class simplex_algorithm_new:
    def __init__(self, setup, dq, coordinates, guess, lateral, settle_time, plot = False, conv_tol = 0.01, max_iter = 20, start_guess = np.array([0.0, 0.0, 0.0]), num_samples = 50):
        self.setup = setup
        self.dq = dq
        self.coordinates = coordinates
        self.guess = guess
        self.lateral = lateral
        self.settle_time = settle_time
        self.plot = plot
        self.conv_tol = conv_tol
        self.max_iter = max_iter
        self.start_guess = start_guess
        self.num_samples = num_samples
    def iterate(self):
        if len(self.coordinates) == 2:
            return self.run_2D()
        elif len(self.coordinates) == 3:
            return self.run_3D()


    def run_2D(self):
        if self.settle_time < 0.05:
            print(
                "Settle time is short. I have a suspicion that this might be harmful to equipment hence this error message.")
            return
        # Lets first try a random guess
        current_volt = np.zeros(4)
        guessAxes = np.setdiff1d([0, 1, 2, 3], self.coordinates)
        current_volt[guessAxes] = self.guess

        L = np.array([4, 4])
        M = np.array([4 + self.lateral, 4])
        H = np.array([4 + self.lateral * np.cos(60 * np.pi / 180), 4 + self.lateral * np.sin(60 * np.pi / 180)])

        # First measure L

        current_volt[self.coordinates] = L
        self.dq.set_volt(current_volt)
        time.sleep(self.settle_time)
        Lval = self.dq.read_input()
        # Measure M
        current_volt[self.coordinates] = M
        self.dq.set_volt(current_volt)
        time.sleep(self.settle_time)
        Mval = self.dq.read_input()

        # Measure H
        current_volt[self.coordinates] = H
        self.dq.set_volt(current_volt)
        time.sleep(self.settle_time)
        Hval = self.dq.read_input()


        for P in [L, M, H]:
            P = np.array(
                [V if V < self.dq.maxV[self.coordinates][i] else self.dq.maxV[self.coordinates][i] for i, V in
                 enumerate(P)])
            P = np.array(
                [V if V > self.dq.minV[self.coordinates][i] else self.dq.minV[self.coordinates][i] for i, V in
                 enumerate(P)])
        conv = 10.0
        L, Lval, M, Mval, H, Hval, _, _ = rearange(L, Lval, M, Mval, H, Hval)
        while conv > self.conv_tol:
            # Rearange such that Lval<Mval<Hval
            # L, Lval, H, Hval, M, Mval = rearange(L, Lval, M, Mval, H, Hval)

            old_min = Lval

            R = self.reflect_triangle(L, M, H)
            current_volt[self.coordinates] = R
            self.dq.set_volt(current_volt)
            time.sleep(self.settle_time)
            Rval = self.dq.read_input()
            if self.plot is True:
                plot_triangle_2d(L, Lval, M, Mval, H, Hval, R, Rval)
            if Rval > Hval:
                E = L + 1.5 * (R - L)
                E = np.array(
                    [V if V < self.dq.maxV[self.coordinates][i] else self.dq.maxV[self.coordinates][i] for i, V in
                     enumerate(E)])
                E = np.array(
                    [V if V > self.dq.minV[self.coordinates][i] else self.dq.minV[self.coordinates][i] for i, V in
                     enumerate(E)])
                current_volt[self.coordinates] = E

                self.dq.set_volt(current_volt)
                time.sleep(self.settle_time)
                Eval = self.dq.read_input()
                if self.plot is True:
                    plot_triangle_2d(L, Lval, M, Mval, H, Hval, R, Rval, E=E, Eval=Eval)

                if Eval > Rval:
                    L = E
                    Lval = Eval
                else:
                    L = R
                    Lval = Rval

            elif Rval > Mval:
                L = R
                Lval = Rval
            elif Rval > Lval:
                Ce = L + (R - L) * 0.75
                Ce = np.array(
                    [V if V < self.dq.maxV[self.coordinates][i] else self.dq.maxV[self.coordinates][i] for i, V in
                     enumerate(Ce)])
                Ce = np.array(
                    [V if V > self.dq.minV[self.coordinates][i] else self.dq.minV[self.coordinates][i] for i, V in
                     enumerate(Ce)])
                current_volt[self.coordinates] = Ce
                self.dq.set_volt(current_volt)
                time.sleep(self.settle_time)
                Ceval = self.dq.read_input()
                if self.plot is True:
                    plot_triangle_2d(L, Lval, M, Mval, H, Hval, R, Rval, Ce=Ce, Ceval=Ceval)
                if Ceval > Rval:
                    L = Ce
                    Lval = Ceval
                else:
                    Cer = self.reflect_triangle(M, Ce, H)
                    current_volt[self.coordinates] = Cer
                    self.dq.set_volt(current_volt)
                    time.sleep(self.settle_time)
                    Cerval = self.dq.read_input()
                    if self.plot is True:
                        plot_triangle_2d(L, Lval, M, Mval, H, Hval, R, Rval, Cer=Cer, Cerval=Cerval)
                    M = Ce
                    Mval = Ceval
                    L = Cer
                    Lval = Cerval
            else:
                Ci = L + (R - L) * 0.25
                Ci = np.array(
                    [V if V < self.dq.maxV[self.coordinates][i] else self.dq.maxV[self.coordinates][i] for i, V in
                     enumerate(P)])
                Ci = np.array(
                    [V if V > self.dq.minV[self.coordinates][i] else self.dq.minV[self.coordinates][i] for i, V in
                     enumerate(P)])
                current_volt[self.coordinates] = Ci
                self.dq.set_volt(current_volt)
                time.sleep(self.settle_time)
                Cival = self.dq.read_input()
                if self.plot is True:
                    plot_triangle_2d(L, Lval, M, Mval, H, Hval, R, Rval, Ci=Ci, Cival=Cival)
                if Cival > Lval:
                    Lval = Cival
                    L = Ci
                else:
                    Cir = self.reflect_triangle(M, Ci, H)
                    current_volt[self.coordinates] = Cir
                    self.dq.set_volt(current_volt)
                    time.sleep(self.settle_time)
                    Cirval = self.dq.read_input()
                    if self.plot is True:
                        plot_triangle_2d(L, Lval, M, Mval, H, Hval, R, Rval, Cir=Cir, Cirval=Cirval)
                    M = Ci
                    Mval = Cival
                    L = Cir
                    Lval = Cirval
            L, Lval, M, Mval, H, Hval, _, _ = rearange(L, Lval, M, Mval, H, Hval)
            conv = np.abs(Hval - Mval)
        current_volt[self.coordinates] = H
        self.dq.set_volt(current_volt)
        time.sleep(self.settle_time)
        Hval = self.dq.read_input()
        return H, Hval

    def run_3D(self):
        #if self.settle_time < 0.05:
        #    print("Settle time is short. I have a suspicion that this might "
        #          "be harmful to equipment hence this error message.")
        #    return
        # Let's first try a random guess
        current_volt = np.zeros(4)

        guessAxes = np.setdiff1d([0, 1, 2, 3], self.coordinates)
        start = self.start_guess

        current_volt[guessAxes] = self.guess
        # Intializing first simplex
        P1 = np.array([start[0], start[1], start[2]])
        P2 = np.array([start[0] + self.lateral, start[1], start[2]])
        P3 = np.array([start[0] + self.lateral * np.cos(60 * np.pi / 180), start[1] + self.lateral * np.sin(60 * np.pi / 180), start[2]])
        P4 = np.zeros(3)
        P4[0:2] = find_center_of_triangle(P1[0:2], P2[0:2], P3[0:2])
        P4[2] = start[2] + np.sqrt(6) / 3 * self.lateral

        P1 = self.set_to_volt_limit(P1)
        P2 = self.set_to_volt_limit(P2)
        P3 = self.set_to_volt_limit(P3)
        P4 = self.set_to_volt_limit(P4)
        # First measure P1
        current_volt[self.coordinates] = P1
        self.setup.set_position([current_volt[0], current_volt[1], current_volt[2]])

        time.sleep(self.settle_time)

        P1val = np.mean(self.dq.read_input_long_continous(self.num_samples))
        # Measure P2
        current_volt[self.coordinates] = P2
        self.setup.set_position([current_volt[0], current_volt[1], current_volt[2]])

        time.sleep(self.settle_time)
        P2val = np.mean(self.dq.read_input_long_continous(self.num_samples))

        # Measure P3
        current_volt[self.coordinates] = P3
        self.setup.set_position([current_volt[0], current_volt[1], current_volt[2]])

        time.sleep(self.settle_time)
        P3val = np.mean(self.dq.read_input_long_continous(self.num_samples))

        # Measure P4
        current_volt[self.coordinates] = P4
        self.setup.set_position([current_volt[0], current_volt[1], current_volt[2]])

        time.sleep(self.settle_time)
        P4val = np.mean(self.dq.read_input_long_continous(self.num_samples))

        conv = 10.0
        P1, P1val, P2, P2val, P3, P3val, P4, P4val = rearange(P1, P1val, P2, P2val, P3, P3val, P4, P4val)
        k = 0
        # while conv > self.conv_tol and k < self.max_iter:
        while P4val < 0.03 and k < self.max_iter:
            P4val_old = P4val
            old_min = P1val
            R = self.reflect_triangle(P1, P2, P3, P4)
            current_volt[self.coordinates] = R
            self.setup.set_position([current_volt[0], current_volt[1], current_volt[2]])

            time.sleep(self.settle_time)
            Rval = np.mean(self.dq.read_input_long_continous(self.num_samples))
            if self.plot is True:
                plot_triangle_3d(P1, P1val, P2, P2val, P3, P3val, P4, P4val, R, Rval)
            if Rval > P4val:  # If R is larger than P4val we make another point even returher out
                E = P1 + 1.5 * (R - P1)
                E = np.array(
                    [V if V < self.dq.maxV[self.coordinates][i] else self.dq.maxV[self.coordinates][i] for i, V in
                     enumerate(E)])
                E = np.array(
                    [V if V > self.dq.minV[self.coordinates][i] else self.dq.minV[self.coordinates][i] for i, V in
                     enumerate(E)])
                current_volt[self.coordinates] = E
                self.setup.set_position([current_volt[0], current_volt[1], current_volt[2]])

                time.sleep(self.settle_time)
                Eval = np.mean(self.dq.read_input_long_continous(self.num_samples))
                if Eval > Rval:
                    P1 = E
                    P1val = Eval
                else:
                    P1 = R
                    P1val = Rval
                if self.plot is True:
                    plot_triangle_3d(P1, P1val, P2, P2val, P3, P3val, P4, P4val, R, Rval, E=E, Eval=Eval)
            elif Rval > P3val:  # TODO so this step I am unclear of. Make it simple for now
                P1 = P2  # TODO i Implemented this step
                P1val = P2val

                P2 = R
                P2val = Rval
            elif Rval > P2val:  # ALso this step is something I made up, to see if we can push it a bit more
                P1 = R
                P1val = Rval
            elif Rval > P1val:  # Here we move to exterior towards R
                Ce = P1 + (R - P1) * 2 / 3
                Ce = np.array(
                    [V if V < self.dq.maxV[self.coordinates][i] else self.dq.minV[self.coordinates][i] for i, V in
                     enumerate(Ce)])
                Ce = np.array(
                    [V if V > self.dq.minV[self.coordinates][i] else self.dq.minV[self.coordinates][i] for i, V in
                     enumerate(Ce)])
                current_volt[self.coordinates] = Ce
                self.setup.set_position([current_volt[0], current_volt[1], current_volt[2]])

                time.sleep(self.settle_time)
                Ceval = np.mean(self.dq.read_input_long_continous(self.num_samples))
                if self.plot is True:
                    plot_triangle_3d(P1, P1val, P2, P2val, P3, P3val, P4, P4val, R, Rval, Ce=Ce, Ceval=Ceval)
                if Ceval > Rval:
                    P1 = Ce
                    P1val = Ceval
                else:
                    # Reflect P2 in Ce, P3, P4
                    Cer = self.reflect_triangle(P2, Ce, P3, P4)
                    current_volt[self.coordinates] = Cer
                    self.setup.set_position([current_volt[0], current_volt[1], current_volt[2]])

                    time.sleep(self.settle_time)
                    Cerval = np.mean(self.dq.read_input_long_continous(self.num_samples))
                    if self.plot is True:
                        plot_triangle_3d(P1, P1val, P2, P2val, P3, P3val, P4, P4val, R, Rval, Cer=Cer, Cerval=Cerval,
                                         Ce=Ce, Ceval=Ceval)
                    P2 = Ce
                    P2val = Ceval
                    P1 = Cer
                    P1val = Cerval
            else:
                Ci = P1 + (R - P1) * 1 / 3  # Make interior point
                Ci = np.array(
                    [V if V < self.dq.maxV[self.coordinates][i] else self.dq.maxV[self.coordinates][i] for i, V in
                     enumerate(Ci)])
                Ci = np.array(
                    [V if V > self.dq.minV[self.coordinates][i] else self.dq.minV[self.coordinates][i] for i, V in
                     enumerate(Ci)])
                current_volt[self.coordinates] = Ci
                self.setup.set_position([current_volt[0], current_volt[1], current_volt[2]])

                time.sleep(self.settle_time)
                Cival = np.mean(self.dq.read_input_long_continous(self.num_samples))
                if self.plot is True:
                    plot_triangle_3d(P1, P1val, P2, P2val, P3, P3val, P4, P4val, R, Rval, Ci=Ci, Cival=Cival)
                if Cival > P1val:
                    P1val = Cival
                    P1 = Ci
                else:
                    # Otherwise we reflect P2 in triangle spanned by Ci, P3 and P4
                    Cir = self.reflect_triangle(P2, Ci, P3, P4)
                    current_volt[self.coordinates] = Cir
                    self.setup.set_position([current_volt[0], current_volt[1], current_volt[2]])

                    time.sleep(self.settle_time)
                    Cirval = np.mean(self.dq.read_input_long_continous(self.num_samples))
                    if self.plot is True:
                        plot_triangle_3d(P1, P1val, P2, P2val, P3, P3val, P4, P4val, R, Rval, Cir=Cir, Cirval=Cirval,
                                         Ci=Ci, Cival=Cival)
                    P2 = Ci
                    P2val = Cival
                    P1 = Cir
                    P1val = Cirval
            P1, P1val, P2, P2val, P3, P3val, P4, P4val = rearange(P1, P1val, P2, P2val, P3, P3val, P4, P4val)
            conv = np.abs(P4val - P3val)
            if P4val == P4val_old:
                k += 1
            else:
                k = 0
            
        current_volt[self.coordinates] = P4
        self.setup.set_position([current_volt[0], current_volt[1], current_volt[2]])

        time.sleep(self.settle_time)
        P4val = np.mean(self.dq.read_input_long_continous(self.num_samples))
        
        return P4, P4val

    def reflect_triangle(self, P1, P2, P3,
                         P4=None):  # Make sure that the points are put in right order. It reflects R1 in the plane of the triangle spanned by the other points
        if P4 is None:
            R = P3 + (P2 - P1)
        else:
            R = P1 - 2 * (P1 - find_center_of_triangle(P2, P3, P4))

        R = np.array([V if V < self.dq.maxV[self.coordinates][i] else self.dq.maxV[self.coordinates][i] for i, V in
                      enumerate(R)])
        R = np.array([V if V > self.dq.minV[self.coordinates][i] else self.dq.minV[self.coordinates][i] for i, V in
                      enumerate(R)])
        return R
    def set_to_volt_limit(self, P):
        P = np.array([V if V < self.dq.maxV[self.coordinates][i] else self.dq.maxV[self.coordinates][i] for i, V in
                      enumerate(P)])
        P = np.array([V if V > self.dq.minV[self.coordinates][i] else self.dq.minV[self.coordinates][i] for i, V in
                      enumerate(P)])
        return P