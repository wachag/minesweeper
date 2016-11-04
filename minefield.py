﻿import sys
import numpy
import random
import copy
import itertools
import time
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt

SIZE = 8
NMINES = 10


def index(x, y):
    return y * SIZE + x


def coords(idex):
    return idex % SIZE, int(idex / SIZE)


def generateminefield(N, Nmines):
    field = -numpy.ones(N * N, numpy.int8)
    mines = [x0 for x0 in range(0, N * N)]
    random.shuffle(mines)
    return field, mines[0:Nmines]

def solveboard(board, mines):
    solution=copy.deepcopy(board)
    for cell in range(0,len(board)):
        if cell in mines:
            solution[cell]=-1
        else:
            neighbours=0;
            (x,y)=coords(cell)
            for ic in range(-1, 2):
                for jc in range(-1, 2):
                    if x + ic >= 0 and x + ic < SIZE and y + jc >= 0 and y + jc < SIZE:
                        if index(x + ic, y + jc) in mines:
                            if not (ic == 0 and jc == 0):
                                neighbours += 1

            solution[cell] = neighbours
    return solution

def generateguess(board, N):
    avail = map(lambda val: None if (val != -1) else True, board)
    it = itertools.compress(range(0, N * N), avail)
    l = list(it)
    return random.choice(l)


def generatescore(gamestate, iter, guess):
    if guess in gamestate[1]:  # mine
        return True, -iter * (gamestate[0].size)
    numcovered = numpy.count_nonzero(gamestate[0] + 1)
    if gamestate[0][guess] != -1:
        return False, -iter * iter * numcovered * numcovered  # negative reward for guessing the known
    return False, iter * iter * numcovered  # positive reward for courage


def generatestate(gstate, guess):
    neighbours = 0
    (x, y) = coords(guess)
    newerstate = copy.deepcopy(gstate)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if(x+i>=0 and x+i<SIZE and y+j>=0 and y+j<SIZE):
                if index(x + i, y + j) in gstate[1]:
                    if not (i == 0 and y == 0):
                        neighbours += 1

    newerstate[0][guess] = neighbours

    return newerstate


def plotMat(ax,X):
    #imshow portion
    ax.imshow(X, interpolation='none')
    #text portion
    diff = 1.
    min_val = 0.
    rows = X.shape[0]
    cols = X.shape[1]
    col_array = numpy.arange(min_val, cols, diff)
    row_array = numpy.arange(min_val, rows, diff)
    x, y = numpy.meshgrid(col_array, row_array)
    for col_val, row_val in zip(x.flatten(), y.flatten()):
        c = X[row_val.astype(int),col_val.astype(int)]
        c = ' ' if c < 1 else c
        ax.text(col_val, row_val, c, va='center', ha='center')
    #set tick marks for grid
    ax.set_xticks(numpy.arange(min_val-diff/2, cols-diff/2))
    ax.set_yticks(numpy.arange(min_val-diff/2, rows-diff/2))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(min_val-diff/2, cols-diff/2)
    ax.set_ylim(min_val-diff/2, rows-diff/2)
    ax.grid()
    plt.show()

if __name__ == "__main__":
    plt.ion()

    model = Sequential([
        Dense(SIZE * int(numpy.sqrt(SIZE)), input_dim=SIZE * SIZE),
        Activation('relu'),
        Dense(SIZE * SIZE),
        Activation('softmax'),
        Dense(SIZE),
        Activation('softmax'),
        Dense(SIZE * SIZE),  # outputs: Coordinates
        Activation('softmax'),
    ])

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    if len(sys.argv) > 1:
        print(sys.argv[1])
        model.load_weights(sys.argv[1])
    startepsilon = 1
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)
    x = [a for a in range(1, 2)]
    y = [a * 0 for a in range(1, 2)]
    line1, = ax1.plot(x, y, 'r-')

    ax2 = fig1.add_subplot(212)
    x = [a for a in range(1, 2)]
    y = [a * 0 for a in range(1, 2)]
    line2, = ax2.plot(x, y, 'r-')
    fig2=plt.figure()
    ax3 = fig2.add_subplot(121)
    ax4 = fig2.add_subplot(122)

    games = 100
    for game in range(0, games):
        ax4.clear()
        print("Playing game ", game)

        epsilon = startepsilon
        if startepsilon > 0.1:
            startepsilon -= (1 / (games))
        gamma = 0.7
        epochs = 100
        originalstate = generateminefield(SIZE, NMINES)
        sol=solveboard(originalstate[0], originalstate[1])
        plotMat(ax4,sol.reshape(SIZE, SIZE))
        iter = 0
        for i in range(1, epochs):
            start = time.time()
            gamestate = copy.deepcopy(originalstate)
            status = 1
            print("Teaching ", i)
            iter = 0
            while True:
                qval = model.predict(gamestate[0].reshape(1, SIZE * SIZE))

                if (random.random() < epsilon):
                    guess = generateguess(gamestate[0], SIZE)
                else:
                    qvalcopy = copy.deepcopy(qval[0])
                    minimum = numpy.min(qvalcopy)
                    for c in range(0, SIZE * SIZE):
                        if gamestate[0][c] != -1:
                            qvalcopy[c] = minimum
                    guess = int(numpy.argmax(qvalcopy))

                (died, reward) = generatescore(gamestate, iter, guess)
                if not died:
                    newstate = generatestate(gamestate, guess)
                    newq = model.predict(newstate[0].reshape(1, SIZE * SIZE))
                else:
                    newstate = copy.deepcopy(gamestate)
                    newq = qval  # TODO?

                update = reward + gamma * numpy.max(newq)
                y = numpy.zeros((1, SIZE * SIZE))
                y[:] = qval[:]

                y[0][guess] = update
                model.fit(gamestate[0].reshape(1, SIZE * SIZE), y, batch_size=1, nb_epoch=game * epochs + i, verbose=0)
                gamestate = copy.deepcopy(newstate)
                numcovered = numpy.count_nonzero(gamestate[0] + 1)
                if (numcovered == SIZE * SIZE - NMINES) or died:
                    end = time.time()
                    if not died:
                        print("Try ", i, " of game", game, " won after ", iter, " iterations ", end - start, " seconds")
                    else:
                        print("Try ", i, " of game", game, " lost after ", iter, " iterations, covered ", numcovered,
                              " from ", SIZE * SIZE, " cells, ", end - start, " seconds")
                    line1.set_xdata(numpy.append(line1.get_xdata(), game * epochs + i))
                    line1.set_ydata(numpy.append(line1.get_ydata(), numcovered * 1.0 / (SIZE * SIZE - NMINES)))
                    line2.set_xdata(numpy.append(line2.get_xdata(), game * epochs + i))
                    line2.set_ydata(numpy.append(line2.get_ydata(), iter))
                    if i % 2 == 0:
                        ax1.relim()
                        ax1.autoscale_view()
                        ax2.relim()
                        ax2.autoscale_view()
                        ax3.clear()
                        plotMat(ax3,(gamestate[0]).reshape(SIZE, SIZE))

                        fig1.canvas.draw()
                        fig2.canvas.draw()
                        model.save("minefield" + str(game) + "_" + str(i) + ".h5")
                        plt.pause(1e-6)  # unnecessary, but useful
                    break
                iter = iter + 1
            if epsilon > 0.1:
                epsilon -= (1 / epochs)
