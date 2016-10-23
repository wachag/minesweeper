﻿import sys
import numpy
import random
import copy
import time
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
SIZE = 10
NMINES = 5

def generateminefield(N, Nmines):
    field = -numpy.ones((N, N), numpy.int8)
    mines = [[x0, y0] for x0 in range(1, N) for y0 in range(1, N)]
    random.shuffle(mines)
    return (field, mines[0:Nmines])

def generateguess(N):
    return [random.randrange(1, N), random.randrange(1, N)]

def generatescore(gamestate, iter, guess):
    if guess in gamestate[1]: # mine
        return (True, -iter*(gamestate[0].size * gamestate[0].size))
    numcovered = numpy.count_nonzero(gamestate[0] + 1)
    if gamestate[0][guess[0]][guess[1]] != -1:
        return (False, -iter*iter*numcovered*numcovered)  # negative reward for guessing the known
    return (False, iter*iter*numcovered)  # positive reward for courage


def generatestate(gamestate, guess):
    neighbours = 0
    x = guess[0]
    y = guess[1]
    newstate = copy.deepcopy(gamestate)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if ([x + i, y + j] in gamestate[1]):
                neighbours = neighbours + 1

    newstate[0][guess[0]][guess[1]] = neighbours
    return newstate


if __name__ == "__main__":
    plt.ion()    
    
    model = Sequential([
        Dense(SIZE*int(numpy.sqrt(SIZE)), input_dim=SIZE * SIZE),
        Activation('relu'),
        Dense(SIZE*SIZE),
        Activation('softmax'),
        Dense(SIZE),
        Activation('softmax'),
        Dense(SIZE * SIZE),  # outputs: Coordinates
        Activation('softmax'),
    ])
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    if len(sys.argv)>1:
        print (sys.argv[1])
        model.load_weights(sys.argv[1])
    startepsilon = 1
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    x=[a for a in range(1,2)]
    y=[a*0 for a in range(1,2)]
    line1, = ax1.plot(x,y,'r-')
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    x=[a for a in range(1,2)]
    y=[a*0 for a in range(1,2)]
    line2, = ax2.plot(x,y,'r-')
    games=100
    for game in range(0,games):
        print("Playing game ",game)
    
        epsilon=startepsilon
        if startepsilon > 0.1:
            startepsilon -= (1 / (games))
        gamma = 0.7
        epochs = 1000
        originalstate = generateminefield(SIZE, NMINES)
        iter=0
        for i in range(1, epochs):
            gamestate=copy.deepcopy(originalstate)
            status = 1
            print("Teaching ",i)
            iter=0
            while status == 1:
                if i % 10 == 0:
                    model.save("minefield"+str(game)+"_"+str(i)+".h5")
                qval = model.predict(gamestate[0].reshape(1, SIZE * SIZE))
                if (random.random() < epsilon):
                    guess = generateguess(SIZE)
                else:
                    a = int(numpy.argmax(qval[0]))
                    guess = [ a % SIZE, int(a / SIZE)]
                (died, reward) = generatescore(gamestate, iter, guess)
                if not died:
                    newstate = generatestate(gamestate, guess)
                    newq = model.predict(gamestate[0].reshape(1, SIZE * SIZE))
                else:
                    newstate = gamestate
                    newq = qval # TODO?
		
                update = reward + gamma * numpy.max(newq)
                y = numpy.zeros((1, SIZE * SIZE))
                y[:] = qval[:]
    
                y[0][guess[1] * SIZE + guess[0]] = update
                model.fit(gamestate[0].reshape(1, SIZE * SIZE), y, batch_size=1, nb_epoch=game*epochs+i, verbose=0)
                gamestate = newstate
                numcovered = numpy.count_nonzero(gamestate[0] + 1)
                if (numcovered==SIZE*SIZE-NMINES) or died:
                    if not died:
                        print("Game", game," won after ", iter, " iterations")
                    else:
                        print("Game", game," lost after ", iter, " iterations, covered ",numcovered," from ",SIZE*SIZE," cells.")
                    line1.set_xdata(numpy.append(line1.get_xdata(), game*epochs+i))
                    line1.set_ydata(numpy.append(line1.get_ydata(), numcovered*1.0/(SIZE*SIZE-NMINES)))
                    line2.set_xdata(numpy.append(line2.get_xdata(), game*epochs+i))
                    line2.set_ydata(numpy.append(line2.get_ydata(), iter))
                    ax1.relim()
                    ax1.autoscale_view()
                    fig1.canvas.draw()
                    ax2.relim()
                    ax2.autoscale_view()
                    fig2.canvas.draw()
                    plt.pause(1e-6) #unnecessary, but useful
                    status=0
                iter=iter+1
            if epsilon > 0.1:
                epsilon -= (1/epochs)