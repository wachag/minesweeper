import numpy
import random
import copy
from keras.models import Sequential
from keras.layers import Dense, Activation


def generateminefield(N, Nmines):
    field = -numpy.ones((N, N), numpy.int8)

    # x=numpy.array([i] for i in range(1,N))
    # y=numpy.array([i] for i in range(1,N))
    mines = [[x0, y0] for x0 in range(1, N) for y0 in range(1, N)]
    random.shuffle(mines)
    return (field, mines[0:Nmines])


def generateguess(N):
    return [random.randrange(1, N), random.randrange(1, N)]


def generatescore(gamestate, guess):
    if guess in gamestate[1]:
        print("BOOM")
        return (True, -(gamestate[0].size * gamestate[0].size))
    numcovered = numpy.count_nonzero(gamestate[0] + 1)
    if gamestate[0][guess[0]][guess[1]] != -1:
        return (False, -numcovered)  # negative reward for guessing the known
    return (False, numcovered)


def generatestate(gamestate, guess):
    neighbours = 0
    x = guess[0]
    y = guess[1]
    newstate = copy.copy(gamestate)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if ([x + i, y + j] in gamestate[1]):
                neighbours = neighbours + 1

    newstate[0][guess[0]][guess[1]] = neighbours
    return newstate


SIZE = 6
NMINES = 5

# jutalom: új és nem akna
# büntetés: régi (kicsi), akna(nagy)


model = Sequential([
    Dense(32, input_dim=SIZE * SIZE),
    Activation('relu'),
    Dense(20),
    Activation('softmax'),
    Dense(SIZE * SIZE),  # outputs: Coordinates
    Activation('softmax'),
])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

epsilon = 1
gamma = 0.99
epochs = 10000
originalstate = generateminefield(SIZE, NMINES)
for i in range(1, epochs):
    gamestate=copy.copy(originalstate)
    status = 1
    while status == 1:
        qval = model.predict(gamestate[0].reshape(1, SIZE * SIZE))
        if (random.random() < epsilon):
            guess = generateguess(SIZE)
        else:
            a = int(numpy.argmax(qval[0]))
            guess = [int(a / SIZE), a % SIZE]
        (died, reward) = generatescore(gamestate, guess)
        if not died:
            newstate = generatestate(gamestate, guess)
        else:
            newstate = gamestate
        newq = model.predict(gamestate[0].reshape(1, SIZE * SIZE))
        update = reward + gamma * numpy.max(newq)
        y = numpy.zeros((1, SIZE * SIZE))
        y[:] = qval[:]

        y[0][guess[0] * SIZE + guess[1]] = update
        model.fit(gamestate[0].reshape(1, SIZE * SIZE), y, batch_size=1, nb_epoch=1, verbose=1)
        gamestate = newstate
        numcovered = numpy.count_nonzero(gamestate[0] + 1)
        if (numcovered==SIZE*SIZE-NMINES) or died:
            if not died:
                print("Yeehaw!")
            status=0
    if epsilon > 0.1:
        epsilon -= (1/epochs)