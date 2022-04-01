# test plotting
import matplotlib.pyplot as plt
import numpy as np
import time

def plotFreq():
    lines = open("../out/ctp2.txt").readlines()
    imag = []
    for i in range(len(lines)):
        split = lines[i].split(", ")
        imag.append(split[1])

    npimag = np.array(imag, dtype=float)

    # plt.plot(2 / 8000 * np.abs(npimag[:8000 // 2]))
    plt.plot(np.abs(npimag[:8000 // 2]))

    plt.show()


def ffttest():
    lines = open("../out/sig.txt").readlines()

    npin = np.array(lines, dtype=float)

    tic = time.perf_counter()
    data = np.fft.fft(npin)
    toc = time.perf_counter()
    plt.plot(np.abs(data[:8000 // 2]))

    #plt.show()
    print(toc - tic)


if __name__ == '__main__':
    ffttest()
