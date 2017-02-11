import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Lambda, Flatten

DROP_PROB = 0.3

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_samples, test_samples = train_test_split(train_samples, test_size=0.3)


def generator(samples, batch_size=32):
    """Generates pairs of images and appropriate steering angles.

    Retrieves images from disk in a batch-wise, memory efficient way.

    :param samples: list of file names in disk
    :param batch_size: number of pairs to return each time the generator is called
    :return: batch of image/steering angle pairs
    """
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            # Lists of images and angles for this batch
            images = []
            angles = []
            for batch_sample in batch_samples:
                # Change this directory string in your machine
                name = '/home/agripino/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            images = np.array(images)
            # Trim image to only see section with road
            X_train = images[:, 80:, :, :]
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
test_generator = generator(test_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.,
                 input_shape=(row, col, ch),
                 output_shape=(row, col, ch)))

# Convolution, relu activation, max pooling and dropout
model.add(Conv2D(nb_filter=30, nb_row=5, nb_col=5, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(p=DROP_PROB))

# Convolution, relu activation, max pooling and dropout
model.add(Conv2D(nb_filter=30, nb_row=5, nb_col=5, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(p=DROP_PROB))

# Convolution, relu activation, max pooling and dropout
model.add(Conv2D(nb_filter=60, nb_row=5, nb_col=5, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(p=DROP_PROB))


model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=3)

print(model.evaluate_generator(test_generator, val_samples=1000))
