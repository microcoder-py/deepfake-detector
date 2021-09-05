import tensorflow as tf
import numpy as np
import data_pipeline as dataGen
import time
import matplotlib.pyplot as plt
#<------------------------------ dataset generation and parameter setting ------------------------------>

capture_sec = 1
num_frames = int(capture_sec*60)

BATCH_SIZE = 4
IMG_SIZE = [100, 100]
NUM_EPOCHS = 10
shuffle_buffer = 800

location = "D:/deepfake_database/dfdc_train_part_1"

strTime = time.time()
train_dataset = dataGen.build_dataset(file_loc = "train_files", img_size = IMG_SIZE, shuffle_buffer = shuffle_buffer, batch_size = BATCH_SIZE, capture_sec = capture_sec)
totTime = time.time() - strTime
print(f"Time To Build Train Dataset - {int(totTime // 60)}:{int(totTime % 60)}\n\n")
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

    self.timeCnn1 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(32, activation = 'relu', kernel_size = (2,2)),
        input_shape = (self.batch_size, self.timesteps, self.height, self.width, self.channels)
    )

    self.avgPool1 = tf.keras.layers.TimeDistributed(
      tf.keras.layers.AveragePooling2D((2,2))
    )

    self.reshape = reshape_for_rnn()
    self.lstm = tf.keras.layers.LSTM(16, activation = 'relu')
    self.flat = tf.keras.layers.Flatten()
    self.dense = tf.keras.layers.Dense(1, 'sigmoid')

  def call(self, inputs):
    convolved1 = self.timeCnn1(inputs)
    pooled1 = self.avgPool1(convolved1)
    reshaped = self.reshape(pooled1)
    rnned = self.lstm(reshaped)
    flattened = self.flat(rnned)
    output = self.dense(flattened)

    return output

temp = tf.constant(np.random.randn(BATCH_SIZE, num_frames, IMG_SIZE[0], IMG_SIZE[1], 3), tf.float32)

# <------------------------------ defining distribute methods ------------------------------>
gpus = tf.config.list_logical_devices('GPU')
distributedStrategy = tf.distribute.MirroredStrategy(gpus)

with distributedStrategy.scope():
    detector = deepfakeDetector()

    #these variables will have their states reset on GPU while the training
    #is ongoing, so we need to include them in the strategy as well
    loss = tf.keras.losses.BinaryCrossentropy(from_logits = False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    list_step_loss = []
    list_epoch_loss = []

    #Doing this so the model runs the build function once
    #since I have not compiled or built the model on the gpus
    op = detector(temp)

step_acc = tf.keras.metrics.BinaryAccuracy(threshold = 0.5)
# <------------------------------ running the training ------------------------------>


list_step_acc = []
list_epoch_acc = []

list_epoch_time = []

for epoch in range(NUM_EPOCHS):
    print(f"<---------------- STARTING EPOCH {epoch} ---------------->")
    strTime = time.time()
    list_step_loss = []
    list_step_acc = []
    train_dataset = train_dataset.shuffle(shuffle_buffer)
    for step, data in enumerate(train_dataset):
        images = data[0]
        fake_val = data[1]

        with tf.GradientTape() as tape:
            op = detector(images)
            loss_val = loss(fake_val,op) + tf.constant(1e-8, tf.float32)

        list_step_loss.append(loss_val)

        grads = tape.gradient(loss_val, detector.trainable_weights)
        optimizer.apply_gradients(zip(grads, detector.trainable_weights))

        with tf.device('/CPU:0'):
            step_acc.reset_state()
            acc_val = step_acc(fake_val, op)
            list_step_acc.append(acc_val)

        print(f"Step: {step}, step loss: {loss_val}, step acc: {acc_val}")

    list_epoch_loss.append(tf.math.reduce_mean(list_step_loss))
    list_epoch_acc.append(tf.math.reduce_mean(list_step_acc))
    totTime = time.time() - strTime
    list_epoch_time.append(totTime)

    print(f"\nEpoch Accuracy: {tf.math.reduce_mean(list_step_acc)}")
    print(f"\nEpoch Loss: {tf.math.reduce_mean(list_step_loss)}")
    print(f"\nTime To Finish Epoch {epoch} - {int(totTime // 60)}:{int(totTime % 60)}\n\n")

list_epochs = [i for i in range(NUM_EPOCHS)]

f = plt.figure()
f.set_figwidth(14)
f.set_figheight(10)

plt.plot(list_epochs, list_epoch_acc, label = "acc")
plt.xlabel('epoch')
plt.title('acc per epoch')
plt.grid()
x1, x2, y1, y2 = plt.axis()
plt.axis((0, NUM_EPOCHS, 0, 1))
plt.xticks(list_epochs, list_epochs)
plt.legend()
plt.savefig('charts/acc_per_epoch.png', dpi=300, bbox_inches='tight')

plt.clf()

plt.plot(list_epochs, list_epoch_loss, label = "loss", color = "red")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss per epoch')
plt.legend()
plt.grid()
x1, x2, y1, y2 = plt.axis()
plt.axis((0, NUM_EPOCHS, y1, y2))
plt.xticks(list_epochs, list_epochs)
plt.savefig('charts/loss_per_epoch.png', dpi=300, bbox_inches='tight')

plt.clf()

plt.plot(list_epochs, list_epoch_time, label = "time_in_sec", color = "green")
plt.xlabel('epoch')
plt.ylabel('time in sec')
plt.title('time per epoch')
plt.xticks(list_epochs, list_epochs)
x1, x2, y1, y2 = plt.axis()
plt.axis((0, NUM_EPOCHS, 0, y2))
plt.legend()
plt.grid()
plt.savefig('charts/time_per_epoch.png', dpi=300, bbox_inches='tight')

# Saving the model for inference or evaluation later
detector.save('model', save_format = tf)
