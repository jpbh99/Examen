
#Nombre = Juan Pablo Batista Hernandez

import cv2
import numpy as np
import sys
import os

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

class hough_():
    def __init__(self, bw_edges):
        [self.rows, self.cols] = bw_edges.shape[:2]
        self.center_x = self.cols // 2
        self.center_y = self.rows // 2
        self.theta = np.arange(0, 360, 0.5)
        self.bw_edges = bw_edges

    def standard_HT(self):

        rmax = int(round(0.5 * np.sqrt(self.rows ** 2 + self.cols ** 2)))
        y, x = np.where(self.bw_edges >= 1)

        accumulator = np.zeros((rmax, len(self.theta)))

        for idx, th in enumerate(self.theta):
            r = np.around(
                (x - self.center_x) * np.cos((th * np.pi) / 180) + (y - self.center_y) * np.sin((th * np.pi) / 180))
            r = r.astype(int)
            r_idx = np.where(np.logical_and(r >= 0, r < rmax))
            np.add.at(accumulator[:, idx], r[r_idx[0]], 1)
        return accumulator

    def direct_HT(self, theta_data):

        rmax = int(round(0.5 * np.sqrt(self.rows ** 2 + self.cols ** 2)))
        # y , x = np.where(M >= 0.1)
        y, x = np.where(self.bw_edges >= 1)

        x_ = x - self.center_x
        y_ = y - self.center_y

        th = theta_data[y, x] + np.pi / 2

        hist_val, bin_edges = np.histogram(th, bins=32)
        print('Histogram', hist_val)

        print(np.amin(th), np.amax(th))
        th[y_ < 0] = th[y_ < 0] + np.pi
        print(np.amin(th), np.amax(th))
        accumulator = np.zeros((rmax, len(self.theta)))

        r = np.around(x_ * np.cos(th) + y_ * np.sin(th))
        r = r.astype(int)
        th = np.around(360 * th / np.pi)
        th = th.astype(int)
        th[th == 720] = 0
        print(np.amin(th), np.amax(th))
        r_idx = np.where(np.logical_and(r >= 0, r < rmax))
        np.add.at(accumulator, (r[r_idx[0]], th[r_idx[0]]), 1)
        return accumulator

    def find_peaks(self, accumulator, nhood, accumulator_threshold, N_peaks):
        done = False
        acc_copy = accumulator
        nhood_center = [(nhood[0] - 1) / 2, (nhood[1] - 1) / 2]
        peaks = []
        while not done:
            [p, q] = np.unravel_index(acc_copy.argmax(), acc_copy.shape)
            if acc_copy[p, q] >= accumulator_threshold:
                peaks.append([p, q])
                p1 = p - nhood_center[0]
                p2 = p + nhood_center[0]
                q1 = q - nhood_center[1]
                q2 = q + nhood_center[1]

                [qq, pp] = np.meshgrid(np.arange(np.max([q1, 0]), np.min([q2, acc_copy.shape[1] - 1]) + 1, 1), \
                                       np.arange(np.max([p1, 0]), np.min([p2, acc_copy.shape[0] - 1]) + 1, 1))
                pp = np.array(pp.flatten(), dtype=np.intp)
                qq = np.array(qq.flatten(), dtype=np.intp)

                acc_copy[pp, qq] = 0
                done = np.array(peaks).shape[0] == N_peaks
            else:
                done = True

        return peaks



class Bandera():
    def __init__(self, Imagen):    # Se define el constructor
        self.Imagen = Imagen

    def Colores(self):
        image = cv2.cvtColor(self.Imagen, cv2.COLOR_BGR2RGB)

        n_colors = 4

        # Convert to floats instead of the default 8 bits integer coding. Dividing by
        # 255 is important so that plt.imshow behaves works well on float data (need to
        # be in the range [0-1])
        image = np.array(image, dtype=np.float64) / 255

        # Load Image and transform to a 2D numpy array.
        rows, cols, ch = image.shape
        assert ch == 3
        image_array = np.reshape(image, (rows * cols, ch))

        print("Fitting model on a small sub-sample of the data")
        image_array_sample = shuffle(image_array, random_state=0)[:10000]
        model = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        self.labels = model.predict(image_array)
        centers = model.cluster_centers_

        Lista = list(set(self.labels))
        Cero = Lista.count(0)
        Uno =  Lista.count(1)
        Dos =  Lista.count(2)
        Tres = Lista.count(3)
        N_col = Cero + Uno + Dos + Tres
        print("La cantida de colores es de: ",str(N_col))

    def Porcentaje(self):
        Lista = list(self.labels)
        Long = len(Lista)
        Cero = (Lista.count(0) / Long) * 100
        Uno = (Lista.count(1) / Long) * 100
        Dos = (Lista.count(2) / Long) * 100
        Tres = (Lista.count(3) / Long) * 100

        Porcent = [Cero, Uno, Dos, Tres]

        print("El porcentaje Porcentaje de cada color es: ", str(Porcent))

    def Orientacion(self):
        image = self.Imagen

        high_thresh = 100
        bw_edges = cv2.Canny(image, high_thresh * 0.3, high_thresh, L2gradient=True)

        hough = hough_(bw_edges)
        accumulator = hough.standard_HT()

        acc_thresh = 50
        N_peaks = 10
        nhood = [25, 9]
        peaks = hough.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

        [_, cols] = image.shape[:2]
        image_draw = np.copy(image)

        Vertical = 0
        Horizontal = 0

        for i in range(len(peaks)):
            rho = peaks[i][0]
            theta_ = hough.theta[peaks[i][1]]

            theta_pi = np.pi * theta_ / 180
            theta_ = theta_ - 180
            a = np.cos(theta_pi)
            b = np.sin(theta_pi)
            x0 = a * rho + hough.center_x
            y0 = b * rho + hough.center_y
            c = -rho
            x1 = int(round(x0 + cols * (-b)))
            y1 = int(round(y0 + cols * a))
            x2 = int(round(x0 - cols * (-b)))
            y2 = int(round(y0 - cols * a))

            if np.abs(theta_) == 180 or np.abs(theta_) == 0:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 255, 255], thickness=2)
                Vertical = 1
            elif np.abs(theta_) == 90:
                image_draw = cv2.line(image_draw, (x1, y1), (x2, y2), [0, 0, 255], thickness=2)
                Horizontal = 1

        if Vertical==1 and Horizontal==1:
            Salida = "mixta"
        elif Vertical == 1:
            Salida = "Vertical"
        else:
            Salida = "Horizontal"

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 1280, 720)
        cv2.imshow("Image", image_draw)
        cv2.waitKey(0)

        return Salida

if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    Imagen = cv2.imread(path_file)


    A = Bandera(Imagen)
    A.Colores()
    A.Porcentaje()
    x = A.Orientacion()
    print("La orientacion es : ", str(x))
