import os
import random
import sys
import datetime

import tensorflow as tf
import tqdm
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (multilabel_confusion_matrix, ConfusionMatrixDisplay,
        classification_report, roc_auc_score, roc_curve, f1_score)

# For Colab
# DSET_IMAGES = os.path.join(os.curdir, './dataset/images/')
# DSET_LABELS = os.path.join(os.curdir, '/content/drive/MyDrive/dataset/dataset.csv')

# On prem
DSET_IMAGES = os.path.join(os.curdir, '../dataset/images/')
DSET_LABELS = os.path.join(os.curdir, '../dataset/dataset.csv')

def get_dataset(dset_path):
    """
    Read CSV of image names and labels convert to Pandas DataFrame.

    Params
    ------
    dset_path: string.
        Path to dataset.

    Returns
    -------
    dset: DataFrame.
        Pandas DataFrame of image names and labels
    """
    dset = pd.read_csv(dset_path).iloc[:, 1:]
    dset = dset.sample(frac=1, random_state=42).reset_index(drop=True)
    return dset

def split_dataset(dset):
    """
    Creating train-val-test splits, 80/10/10.

    Params
    ------
    dset: DataFrame.
        Pandas DataFrame of image names and labels

    Returns
    -------
    (X_train, y_train, X_val, y_val, X_test, y_test) - tuple, array-like
        Image paths (X) are returned as DataFrames. Labels (y) are returned as
        tensors.
    """
    global DSET_IMAGES

    X = (DSET_IMAGES + dset['image_name'])
    y = dset.iloc[:, 2:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42, test_size=int(dset.shape[0]*0.1))

    y_train = tf.constant(y_train)
    y_val = tf.constant(y_val)
    y_test = tf.constant(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def _bytes_feature(value):
    """
    Converts serializable objects to a bytes feature for storage in TFRecords.
    Used for storing raw image data and tensors of labels for each example.

    Params
    ------
    value: object
        A serializable Python object

    Returns
    -------
    feature: Feature
        A TensorFlow feature object for an example protobuf.
    """
    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return feature

# Create an example class for the TFRecord protobuf
def create_example(image_string, labels):
    feature = {
        'image_raw': _bytes_feature(image_string.numpy()),
        'labels': _bytes_feature(tf.io.serialize_tensor(labels).numpy()),
    }
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto

def crop_image(img, tol=4):
    """
    Crop black borders around fundus images

    Params
    ------
    img: array-like
        Image data, in an array representation.

    Returns
    -------
    img: array-like
        Cropped image.
    """
    # tol is tolerance, pixels to exclude from the mask
    img = np.asarray(img)
    mask = img > tol
    if img.ndim==3:
        mask = mask.all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    img = img[row_start:row_end,col_start:col_end]
    img = tf.convert_to_tensor(img)
    return img

def create_record(path, X, y):
    """
    
    """
    options = tf.io.TFRecordOptions(compression_type='GZIP')
    y = tf.convert_to_tensor(y)
    with tf.io.TFRecordWriter(path, options=options) as writer:
        for i in range(len(X)):
            try:
                image = tf.io.decode_image(tf.io.read_file(X[i]))
            except tf.errors.InvalidArgumentError as e:
                print(X[i])
                continue
            image = tf.io.decode_image(tf.io.read_file(X[i]))
            image = crop_image(image)
            image = tf.image.resize(image, (244, 244), method='nearest', preserve_aspect_ratio=True)
            image = tf.io.encode_jpeg(image)
            label = y[i]
            tf_example = create_example(image, label)
            writer.write(tf_example.SerializeToString())

# On prem
#dset_train = os.path.join(os.curdir, '../records244/train')
#dset_val = os.path.join(os.curdir, '../records244/dset_val.tfrecord')
#dset_test = os.path.join(os.curdir, '../records244/dset_test.tfrecord')

# Colab
dset_train = os.path.join(os.curdir, './records244/dset_train.tfrecord')
dset_val = os.path.join(os.curdir, './records244/dset_val.tfrecord')
dset_test = os.path.join(os.curdir, './records244/dset_test.tfrecord')


AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.random.set_seed(42)
random.seed(42)
train_dataset = tf.data.TFRecordDataset(dset_train, compression_type='GZIP').shuffle(2048)
val_dataset = tf.data.TFRecordDataset(dset_val, compression_type='GZIP')
test_dataset = tf.data.TFRecordDataset(dset_test, compression_type='GZIP')

def _parse_dataset(example_proto, img_size=[244, 244]):
    image_feature_description = {
        'labels': tf.io.FixedLenFeature([], tf.string),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)
    labels = tf.io.parse_tensor(parsed['labels'], tf.int64)
    labels.set_shape((8,))
    image = tf.io.decode_image(parsed['image_raw']) / 255
    image.set_shape(img_size + [3])
    return image, labels

image_feature_description = {
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'labels': tf.io.FixedLenFeature([], tf.string)
}
train_dataset = train_dataset.map(_parse_dataset, num_parallel_calls=AUTOTUNE).batch(32).prefetch(AUTOTUNE)
val_dataset = val_dataset.map(_parse_dataset, num_parallel_calls=AUTOTUNE).batch(32).prefetch(AUTOTUNE)
test_dataset = test_dataset.map(_parse_dataset, num_parallel_calls=AUTOTUNE).batch(32).prefetch(AUTOTUNE)

# Split a single TF Record into multiple files
def _parse_raw_dataset(example_proto, img_size=[360, 360]):
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)
    labels = tf.io.parse_tensor(parsed['labels'], tf.int64)
    labels.set_shape((8,))
    image = parsed['image_raw']
    return image, labels

def shard_tfrecord(path, name, record):
    options = tf.io.TFRecordOptions(compression_type='GZIP')
    dset = tf.data.TFRecordDataset(record, compression_type='GZIP')
    dset = dset.map(_parse_raw_dataset).batch(1000)
    batch_id = 0
    for batch in dset:
        batch_ds = tf.data.Dataset.from_tensor_slices(batch)
        filename = f'{name}.{batch_id:03d}'
        record_path = os.path.join(path, filename)
        with tf.io.TFRecordWriter(record_path, options=options) as writer:
            for ex in batch_ds:
                tf_example = create_example(*ex)
                writer.write(tf_example.SerializeToString())
        batch_id += 1
        
def augment_image(image):
    image = tf.image.random_flip_left_right(image, seed=42)
    image = tf.image.random_flip_up_down(image, seed=42)
    image = tf.image.random_contrast(image, 0.8, 1.0, seed=42)
    image = tf.image.random_brightness(image, 0.1, seed=42)
    image = tf.minimum(image, 255)
    return image

def oversample_dataset(record, target):
    labels = []
    images = []
    
    augmented_images = []
    augmented_labels = []
    
    for batch in record:
        image = batch[0]
        label = batch[1]
        images.append(image)
        labels.append(label)
        
    images = tf.concat(images, 0)
    labels = tf.concat(labels, 0)
   
    for i in range(8):
        current = labels[:, i] == 1
        current_img = images[current]
        current_labels = labels[current]
        current_number = current_img.shape[0]
        
        # Number of images we need to create
        num_create = target - current_number
        
        if num_create < 0:
            augmented_images.extend(current_img)
            augmented_labels.extend(current_labels)
            continue
        
        ind = 0 
        for j in tqdm.trange(num_create):
            over_img = tf.io.decode_image(current_img[ind])
            over_lab = current_labels[ind]
            over_img = tf.io.encode_jpeg(augment_image(over_img))
            
            augmented_images.append(over_img)
            augmented_labels.append(over_lab)
            ind = min(ind, current_img.shape[0]-1)
            
        augmented_images.extend(current_img)
        augmented_labels.extend(current_labels)
            
    augmented_images = tf.stack(augmented_images)
    augmented_labels = tf.stack(augmented_labels)
    
    tf.random.set_seed(42)
    indices = tf.random.shuffle(tf.range(augmented_images.shape[0]), seed=42)
    augmented_images = tf.gather(augmented_images, indices)
    augmented_labels = tf.gather(augmented_labels, indices)
    
    return augmented_images, augmented_labels
        
def undersample(majority_images, majority_labels, target):
    tf.random.set_seed(42)
    indices = tf.random.shuffle(tf.range(majority_images.shape[0]), seed=42)[:target]
    under_sampled_img = tf.gather(majority_images, indices)
    under_sampled_labels = tf.gather(majority_labels, indices)
    return under_sampled_img, under_sampled_labels

# Callbacks
class ExpoIncreaseLRCallback(keras.callbacks.Callback):
    """
    Exponentially increases the learning rate by a constant factor. Meant to
    be run over a few hundred iterations in one epoch. Stores a history object
    of the loss and the learning rates. The optimal loss is usually about 10x
    lower than when the algorithm diverges (loss shoots up).
    """
    def __init__(self, factor): 
        super().__init__()
        self.loss = []
        self.lr = []
        self.factor = factor
        
    def on_batch_end(self, batch, logs=None):
        loss = logs['loss']
        prev_lr = K.get_value(self.model.optimizer.lr)
        self.lr.append(prev_lr)
        self.loss.append(loss)
        K.set_value(self.model.optimizer.lr, prev_lr * self.factor)

class OneCycleScheduler(keras.callbacks.Callback):
    """
    Learning rate scheduler, implementing a cyclical learning rate (Smith, 2018).
    https://arxiv.org/abs/1803.09820 
    """
    def __init__(self, epoch_size, batch_size, max_lr, max_momentum=0, min_momentum=0):
        super(OneCycleScheduler, self).__init__()
        self.max_lr = max_lr
        self.min_lr = max_lr / 10
        self.progress = 0
        self.iterations = epoch_size // batch_size
        self.max_momentum = max_momentum
        self.min_momentum = min_momentum
        
    def on_epoch_begin(self, epoch, logs=None):
        keras.backend.set_value(self.model.optimizer.lr, self.min_lr)
        keras.backend.set_value(self.model.optimizer.momentum, self.max_momentum)
        self.progress = 0
    
    def on_train_batch_begin(self, batch, logs=None):
        self.progress += 1
        
        # Finding rate of change to halfway through epoch
        half = self.iterations // 2 
        lr_roc = (self.max_lr - self.min_lr) / half
        
        # Increase if first half, else decrease
        if self.progress >= half:
            lr_roc *= -1
        lr = self.model.optimizer.lr
        cur_lr = lr + lr_roc
        keras.backend.set_value(self.model.optimizer.lr, cur_lr)
        
        # Finding rate of change for momentum
        momentum_roc = -((self.max_momentum - self.min_momentum) / half)
        
        # Decrease if first half, else decrease
        if self.progress >= half:
            momentum_roc *= -1
        momentum = self.model.optimizer.momentum
        cur_momentum = momentum + momentum_roc
        keras.backend.set_value(self.model.optimizer.momentum, cur_momentum)

def make_vgg_net():
    """
    Implementation of the 2015 ILSVRC second place VGGNet-19 architecture.
    Note that the placement of the batch normalization layers in front of
    the activation was purposeful. There is some debate regarding whether
    or not the placement of BN before or after ReLU has any large effect
    on performance.

    From 
    """
    augmentation = keras.models.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        keras.layers.experimental.preprocessing.RandomFlip('vertical'),
    ])
    normalization = keras.layers.experimental.preprocessing.Normalization()
    for batch in train_dataset.take(3):
        img = batch[0]
        normalization.adapt(img)

    model = keras.models.Sequential([
        augmentation,
        normalization,
        keras.layers.Conv2D(64, 3, padding="same", input_shape=[244, 244, 3]),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, 3, padding="same"),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, padding="same"),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(128, 3, padding="same"),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),
        keras.layers.Conv2D(256, 3, padding="same"),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(256, 3, padding="same"),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(256, 3, padding="same"),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(256, 3, padding="same"),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),
        keras.layers.Conv2D(512, 3, padding="same"),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(512, 3, padding="same"),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(512, 3, padding="same"),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(512, 3, padding="same"),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),
        keras.layers.Conv2D(512, 3, padding="same"),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(512, 3, padding="same"),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(512, 3, padding="same"),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(512, 3, padding="same"),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),
        keras.layers.Flatten(),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(4096, kernel_initializer='he_normal'),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(4096, kernel_initializer='he_normal'),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(1000, kernel_initializer='he_normal'),
        keras.layers.Activation('relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(8, activation="sigmoid")
    ])
    return model

def train_nadam_vgg_19():
    K.clear_session()
    tf.random.set_seed(42)
    random.seed(42)
    model = make_vgg_net()
    model.compile(optimizer=keras.optimizers.Nadam(lr=3e-4),
            metrics=['binary_accuracy', 'AUC'], loss='binary_crossentropy')
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = keras.callbacks.TensorBoard(logdir)
    model_checkpoint_cb = keras.callbacks.ModelCheckpoint(
            '/content/drive/MyDrive/models/custom_vggnet19.h5',
            monitor='val_loss', save_best_only=True, save_freq='epoch')
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
    csv_logger_cb =tf.keras.callbacks.CSVLogger(
            '/content/drive/MyDrive/learning_curves/custom_vggnet19.csv',
            separator=",", append=False)

    history = model.fit(
            train_dataset, epochs=1000, validation_data=val_dataset,
            callbacks=[tensorboard_cb, model_checkpoint_cb,
                early_stopping_cb, csv_logger_cb])

    return model

# Code for model predictions
def create_confusion_matrix(y_true, y_pred, names):
    confusion = multilabel_confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.tight_layout()
    axes = axes.ravel()

    for ind, ax in enumerate(axes):
        confusion_display = ConfusionMatrixDisplay(confusion[ind], display_labels=[0, 1])
        confusion_display.plot(ax=ax, values_format='.5g')
        ax.set_title(names[ind])

def _to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

def get_thresholds(model, dataset, samples=1):
    y_prob = np.stack([model.predict(dataset) for sample in range(samples)]).mean(axis=0)
    y_true = []
    for ind, batch in enumerate(dataset):
        y_true.append(batch[1])
    
    y_true = tf.concat(y_true, 0)
    thresholds = np.arange(0, 1, 0.001)
    class_thresholds = []

    for i in range(8):
        scores = [f1_score(y_true[:, i], _to_labels(y_prob[:, i], t)) for t in thresholds] 
        ix = np.argmax(scores)
        class_thresholds.append(thresholds[ix])

    return class_thresholds

def get_predictions(model, class_thresholds, dataset, samples=1):
    y_prob = np.stack([model.predict(dataset) for sample in range(samples)]).mean(axis=0)
    y_true = []
    for ind, batch in enumerate(dataset):
        y_true.append(batch[1])
    
    y_true = tf.concat(y_true, 0)
    y_pred = []
    for ind, thresh in enumerate(class_thresholds):
        class_pred = np.where(y_prob[:, ind] > thresh, 1, 0)
        y_pred.append(tf.reshape(class_pred, (-1, 1)))
    y_pred = tf.concat(y_pred, 1)
    return y_true, y_pred




# Custom VGGNet error analysis
model = keras.models.load_model('/content/drive/MyDrive/models/custom_vggnet19.h5')
thresholds = get_thresholds(model, val_dataset)        
y_true, y_pred = get_predictions(model, thresholds, test_dataset)
print(classification_report(y_true, y_pred))
print(roc_auc_score(y_true, y_pred))
create_confusion_matrix(y_true, y_pred, dset.columns[2:])




# VGGNet, same architecture, instead using SGD
K.clear_session()
tf.random.set_seed(42)
random.seed(42)

# Finding the optimial learning rate
expo_lr = ExpoIncreaseLRCallback(1.032)
model = make_vgg_net()
model.compile(optimizer=keras.optimizers.SGD(lr=0.0001), metrics=['binary_accuracy', 'AUC'], loss='binary_crossentropy')
model.fit(train_dataset, epochs=1, callbacks=[expo_lr])




plt.plot(expo_lr.lr, expo_lr.loss)
plt.xscale('log')


# 0.11 seems to be a good learning rate. We can use this as the maximum learning rate for our cyclical learning rate scheduler.



model = make_vgg_net()

# Callbacks
one_cycle_scheduler = OneCycleScheduler(11020, 32, 0.11)
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = keras.callbacks.TensorBoard(logdir)
model_checkpoint_cb = keras.callbacks.ModelCheckpoint('/content/drive/MyDrive/models/SGD_vggnet19.h5', monitor='val_loss', save_best_only=True, save_freq='epoch')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
csv_logger_cb =tf.keras.callbacks.CSVLogger('/content/drive/MyDrive/learning_curves/SGD_vggnet19.csv', separator=",", append=False)

optimizer = keras.optimizers.SGD(lr=0.11)
model.compile(optimizer=optimizer, metrics=['binary_accuracy', 'AUC'], loss='binary_crossentropy')
history = model.fit(train_dataset, epochs=1000, validation_data=val_dataset,
                    callbacks=[one_cycle_scheduler, tensorboard_cb, model_checkpoint_cb, early_stopping_cb, csv_logger_cb])




class SEResidualUnit(keras.layers.Layer):
    """
    The residual units with skip connections and SE-blocks. Based on
    Hu et al, (2017)

    https://arxiv.org/abs/1709.01507
    """
    def __init__(self, filters, ratio=8, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.filters = filters
        self.strides = strides

        # The main layers of the residual unit, two convolutional layers,
        # one with a stride of one and batch normalization layers
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides,
                                padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1,
                                padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
        ]

        # Defining the skip connection and the convolutional layer the inputs
        # have to pass through so as to ensure the shapes of added inputs are
        # the same as those going through the main layer
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides,
                                    padding='same', use_bias=False),
                keras.layers.BatchNormalization()
            ]

        # SE Block
        self.ratio = ratio
        self.se_block = [
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(filters//ratio, activation='relu', kernel_initializer='he_normal', use_bias=False),
            keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False),
        ]

    def call(self, inputs):
        """
        Adding the inputs from the skip connection to the outputs and passing
        output of Residual Unit to SE block.
        """
        # Residual unit
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)

        # SE Block
        X = Z
        for layer in self.se_block:
            X = layer(X)
        Z = keras.layers.multiply([X, Z]) 

        # Skip connection
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)

        out = self.activation(Z + skip_Z)
        return out

    def get_config(self):
        activation = self.activation
        filters = self.filters
        strides = self.strides
        ratio = self.ratio
        base_config = super().get_config()
        return {**base_config, 'filters': filters, 'strides': strides,
        'activation': keras.activations.serialize(activation), 'ratio': ratio}


def create_resnet():
    """
    SE-ResNet-34 architecture

    """
    augmentation = keras.models.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        keras.layers.experimental.preprocessing.RandomFlip('vertical'),
    ])
    model = keras.models.Sequential()
    model.add(augmentation)
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=(244, 244, 3),
                        padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(2))
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if prev_filters == filters else 2
        model.add(SEResidualUnit(filters, strides=strides, activation='relu'))
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1000, kernel_initializer='he_normal'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(8, activation='sigmoid'))
    return model

def find_sdg_se_resnet_lr():
    K.clear_session()
    tf.random.set_seed(42)
    random.seed(42)
    one_cycle = OneCycleScheduler(11020, 32, 0.4)
    precision = keras.metrics.Precision()
    recall = keras.metrics.Recall()
    expo_lr = ExpoIncreaseLRCallback(1.05)
    model = create_resnet()
    model.compile(optimizer=keras.optimizers.SGD(lr=0.4), loss='binary_crossentropy', metrics=['binary_accuracy', precision, recall, 'AUC'])
    history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[one_cycle])
    return history



def train_sgd_se_resnet():
    # Training SE-Resnet-34 model
    K.clear_session()
    tf.random.set_seed(42)
    random.seed(42)
    model = create_resnet()

    one_cycle = OneCycleScheduler(11020, 32, 0.4)
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = keras.callbacks.TensorBoard(logdir)
    model_checkpoint_cb = keras.callbacks.ModelCheckpoint(
        '/content/drive/MyDrive/models/SE_resnet34.h5', monitor='val_loss', save_best_only=True, save_freq='epoch')
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
    csv_logger_cb = tf.keras.callbacks.CSVLogger('/content/drive/MyDrive/learning_curves/SE_resnet34_no_class_weights.csv', separator=",", append=False)

    precision = keras.metrics.Precision()
    recall = keras.metrics.Recall()

    model.compile(optimizer=keras.optimizers.SGD(lr=0.4),
            loss='binary_crossentropy',
            metrics=['binary_accuracy', precision, recall, 'AUC'])

    history = model.fit(train_dataset, epochs=1000, validation_data=val_dataset,
                        callbacks=[one_cycle, model_checkpoint_cb, 
                            early_stopping_cb, csv_logger_cb, tensorboard_cb])
    return model

def train_nadam_se_resnet():
    # Training SE-Resnet-34 model with Nadam
    K.clear_session()
    tf.random.set_seed(42)
    random.seed(42)
    model = create_resnet()

    # Callbacks
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = keras.callbacks.TensorBoard(logdir)
    model_checkpoint_cb = keras.callbacks.ModelCheckpoint(
        '/content/drive/MyDrive/models/Nadam_SE_resnet34.h5',
        monitor='val_loss', save_best_only=True, save_freq='epoch')
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=30)
    csv_logger_cb = tf.keras.callbacks.CSVLogger(
            '/content/drive/MyDrive/learning_curves/SE_resnet34_Nadam.csv',
            separator=",", append=True)

    # Metrics
    precision = keras.metrics.Precision()
    recall = keras.metrics.Recall()

    # Compile model
    model.compile(optimizer=keras.optimizers.Nadam(lr=3e-4),
            loss='binary_crossentropy',
            metrics=['binary_accuracy', precision, recall, 'AUC'])

    # Model training
    history = model.fit(train_dataset, epochs=1000, validation_data=val_dataset,
                        callbacks=[model_checkpoint_cb, early_stopping_cb, 
                            csv_logger_cb, tensorboard_cb])
    return model


def train_class_weights_se_resnet():
    # Computing class weights
    totals = tf.reduce_sum(y_train, axis=0).numpy()
    labels = []
    for ind, val in enumerate(totals):
        labels.extend([ind] * val)
    class_weights = compute_class_weight('balanced', np.arange(8), labels)
    class_weights = dict(enumerate(class_weights))

    # Training a class weighted SE_Resnet
    K.clear_session()
    tf.random.set_seed(42)
    random.seed(42)

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = keras.callbacks.TensorBoard(logdir)
    model_checkpoint_cb = keras.callbacks.ModelCheckpoint(
            '/content/drive/MyDrive/models/SE_resnet34_class_weights.h5',
            monitor='val_loss', save_best_only=True, save_freq='epoch')
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
    csv_logger_cb = tf.keras.callbacks.CSVLogger(
            '/content/drive/MyDrive/learning_curves/SE_resnet34_class_weights.csv',
            separator=",", append=False)
    precision = keras.metrics.Precision()
    recall = keras.metrics.Recall()
    model.compile(optimizer=keras.optimizers.Nadam(lr=3e-4), 
            metrics=['binary_accuracy', precision, recall, 'AUC'],
            loss='binary_crossentropy')
    history = model.fit(train_dataset, epochs=1000, validation_data=val_dataset,
                        callbacks=[tensorboard_cb, model_checkpoint_cb, 
                                   early_stopping_cb, csv_logger_cb], 
                        class_weight=class_weights)
    return model

def train_oversampled_se_resnet_34():
    # Testing oversampling, creating balanced classes with copies of minority
    # classes with data augmentation
    dset_train_oversample = os.path.join(os.curdir, './records_over1000/dset_train.tfrecord')
    train_dataset_oversample = tf.data.TFRecordDataset(dset_train_oversample, compression_type='GZIP').shuffle(2048)
    train_dataset_oversample = train_dataset_oversample.map(_parse_dataset, num_parallel_calls=AUTOTUNE).batch(32).prefetch(AUTOTUNE)

    # Training an SE-Resnet-34 on the oversampled dataset
    K.clear_session()
    tf.random.set_seed(42)
    random.seed(42)
    one_cycle = OneCycleScheduler(11020, 32, 0.4)
    precision = keras.metrics.Precision()
    recall = keras.metrics.Recall()
    expo_lr = ExpoIncreaseLRCallback(1.035)
    model = create_resnet()
    model.compile(optimizer=keras.optimizers.Nadam(lr=3e-4), loss='binary_crossentropy', metrics=['binary_accuracy', precision, recall, 'AUC'])
    history = model.fit(train_dataset_oversample, epochs=10, validation_data=val_dataset)
    return model


if __name__ == '__main__':
    pass
