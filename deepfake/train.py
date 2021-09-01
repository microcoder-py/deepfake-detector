import tensorflow as tf
import numpy as np
import data_pipeline as dataGen

#<------------------------------ dataset generation and parameter setting ------------------------------>

capture_sec = 1
num_frames = int(capture_sec*60)

BATCH_SIZE = 16
IMG_SIZE = [180, 180]
NUM_EPOCHS = 20
train_dataset = dataGen.build_dataset(capture_sec = capture_sec, dataset_type = 'train', batch_size = BATCH_SIZE, img_size = IMG_SHAPE)

# <------------------------------ model building ------------------------------>

#this layer is required for reshaping the inputs passed from the convolution ops
#to the LSTM layer which only accepts 1D inputs in time sequence.
#Keras can handle this automatically in the sequential model but I find defining
#it clearly helps with modifications later
class reshape_for_rnn(tf.keras.layers.Layer):
    def __init__(self):
        super(reshape_for_rnn, self).__init__()

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.timesteps = input_shape[1]
        self.height = input_shape[2]
        self.width = input_shape[3]
        self.channels = input_shape[4]

    def call(self, inputs):
        return tf.reshape(inputs, (self.batch_size, self.timesteps, self.height*self.width*self.channels))

#defining the model, a CNN-LSTM
#CNN layer tracks the different features in the image sequence, LSTM performs
#time sequence analysis
class deepfakeDetector(tf.keras.Model):
  def __init__(self):
    super(deepfakeDetector, self).__init__()

  def build(self, input_shape):

    self.batch_size = input_shape[0]
    self.timesteps = input_shape[1]
    self.height = input_shape[2]
    self.width = input_shape[3]
    self.channels = input_shape[4]

    self.timeCnn = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(32, kernel_size = (3,3)),
        input_shape = (self.batch_size, self.timesteps, self.height, self.width, self.channels)
    )

    self.maxPool = tf.keras.layers.TimeDistributed(
      tf.keras.layers.MaxPooling2D((2,2))
    )

    self.reshape = reshape_for_rnn()
    self.lstm = tf.keras.layers.LSTM(16)
    self.flat = tf.keras.layers.Flatten()
    self.dense = tf.keras.layers.Dense(1, 'sigmoid')

  def call(self, inputs):
    convolved = self.timeCnn(inputs)
    pooled = self.maxPool(convolved)
    reshaped = self.reshape(pooled)
    rnned = self.lstm(reshaped)
    flattened = self.flat(rnned)
    output = self.dense(flattened)

    return output

temp = tf.constant(np.random.randn(BATCH_SIZE, num_frames, IMG_SIZE[0], IMG_SIZE[1], 3), tf.float32)

# <------------------------------ defining distribute methods ------------------------------>
gpus = tf.config.list_logical_devices('GPU')
distributionStrategy = tf.distribute.MirroredStrategy(gpus)

with distributedStrategy.scope():
    detector = deepfakeDetector()

    #Doing this so the model runs the build function once
    #since I have not compiled or built the model on the gpus
    op = detector(temp)

# <------------------------------ running the training ------------------------------>

loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
acc = tf.keras.metrics.Accuracy()

for epoch in range(NUM_EPOCHS):
    print(f"<---------------- STARTING EPOCH {epoch} ---------------->")
    strTime = time.time()

    for step, data in enumerate(train_dataset):
        images = data[0]
        fake_val = data[1]

         with tf.GradientTape() as tape:
             op = detector(images)
             loss_val = loss(fake_val, op)

        grads = tape.gradient(loss_val, detector.trainable_weights)
        optimizer.apply_gradients(zip(grads, detector.trainable_weights))
        acc_val = acc(fake_val, op)

        if (step % 500 == 0):
            print(f"Step: {step}, loss: {loss_val}, acc: {acc_val}")

    totTime = time.time() - strTime
    print(f"Time To Finish Epoch {epoch}: {int(totTime // 60)}:{int(totTime % 60)}\n\n")

# Saving the model for inference or evaluation later
detector.save('model', save_format = tf)
