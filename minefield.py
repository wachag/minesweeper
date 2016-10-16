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

def generatescore(gamestate, guess):
    if guess in gamestate[1]:
        return (True, -(gamestate[0].size * gamestate[0].size))
    numcovered = numpy.count_nonzero(gamestate[0] + 1)
    if gamestate[0][guess[0]][guess[1]] != -1:
        return (False, -numcovered)  # negative reward for guessing the known
    return (False, numcovered)


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
    
plt.ion()    




model = Sequential([
    Dense(SIZE*int(numpy.sqrt(SIZE)), input_dim=SIZE * SIZE),
    Activation('relu'),
    Dense(SIZE),
    Activation('softmax'),
    Dense(SIZE),
    Activation('softmax'),
    Dense(SIZE * SIZE),  # outputs: Coordinates
    Activation('softmax'),
])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
startepsilon = 1
for game in range(1,100):
    print("Playing game ",game)
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


    epsilon=startepsilon
    if startepsilon > 0.1:
        startepsilon -= (1 / game)
    gamma = 0.9
    epochs = 1000
    originalstate = generateminefield(SIZE, NMINES)
    iter=0
    for i in range(1, epochs):
        gamestate=copy.deepcopy(originalstate)
        status = 1
        print("Teaching ",i)
        iter=0
        while status == 1:
            qval = model.predict(gamestate[0].reshape(1, SIZE * SIZE))
            if (random.random() < epsilon):
                guess = generateguess(SIZE)
            else:
                a = int(numpy.argmax(qval[0]))
                guess = [ a % SIZE, int(a / SIZE)]
            (died, reward) = generatescore(gamestate, guess)
            if not died:
                newstate = generatestate(gamestate, guess)
            else:
                newstate = gamestate
            newq = model.predict(gamestate[0].reshape(1, SIZE * SIZE))
            update = reward + gamma * numpy.max(newq)
            y = numpy.zeros((1, SIZE * SIZE))
            y[:] = qval[:]

            y[0][guess[1] * SIZE + guess[0]] = update
            model.fit(gamestate[0].reshape(1, SIZE * SIZE), y, batch_size=1, nb_epoch=i, verbose=0)
            gamestate = newstate
            numcovered = numpy.count_nonzero(gamestate[0] + 1)
            if (numcovered==SIZE*SIZE-NMINES) or died:
                if not died:
                    print("Game", game," won after ", iter, " iterations")
                else:
                    print("Game", game," lost after ", iter, " iterations, covered ",numcovered," from ",SIZE*SIZE," cells.")
                line1.set_xdata(numpy.append(line1.get_xdata(), i))
                line1.set_ydata(numpy.append(line1.get_ydata(), numcovered*1.0/(SIZE*SIZE-NMINES)))
                line2.set_xdata(numpy.append(line2.get_xdata(), i))
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