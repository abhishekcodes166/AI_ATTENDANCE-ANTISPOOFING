"""FaceNet architecture built and trained entirely from scratch.

A lightweight FaceNet (Schroff et al., 2015, "FaceNet: A Unified Embedding
for Face Recognition and Clustering") in the style of the paper's NN1
(Zeiler & Fergus) network, sized so it can actually be trained from random
initialization on a dataset collected by this application.

Guarantees required by the project:
  * Random weight initialization — no pretrained weights of any kind
  * Batch Normalization after every convolution
  * ReLU activations
  * 128-dimensional embedding output
  * L2 normalization of the embedding (UnitNormalization layer)
"""

import keras
from keras import layers

import config


def _conv_block(x, filters, kernel=3, strides=1, name=None):
    """Conv -> BatchNorm -> ReLU (randomly initialized)."""
    x = layers.Conv2D(
        filters, kernel, strides=strides, padding="same", use_bias=False,
        kernel_initializer="he_normal", name=None if name is None else name + "_conv",
    )(x)
    x = layers.BatchNormalization(name=None if name is None else name + "_bn")(x)
    x = layers.ReLU(name=None if name is None else name + "_relu")(x)
    return x


def build_facenet(input_size=None, embedding_size=None):
    """Build the embedding network f(x) -> R^128 with ||f(x)||_2 = 1."""
    input_size = input_size or config.FACE_IMAGE_SIZE
    embedding_size = embedding_size or config.EMBEDDING_SIZE

    inputs = keras.Input(shape=(input_size, input_size, 3), name="face_input")

    # Stem (NN1-style 7x7 entry convolution)
    x = _conv_block(inputs, 64, kernel=7, strides=2, name="stem")      # 48x48
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)           # 24x24

    # Block 2
    x = _conv_block(x, 64, kernel=1, name="b2_reduce")
    x = _conv_block(x, 128, kernel=3, name="b2")
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)           # 12x12

    # Block 3
    x = _conv_block(x, 128, kernel=1, name="b3_reduce")
    x = _conv_block(x, 256, kernel=3, name="b3a")
    x = _conv_block(x, 256, kernel=3, name="b3b")
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)           # 6x6

    # Block 4
    x = _conv_block(x, 256, kernel=1, name="b4_reduce")
    x = _conv_block(x, 512, kernel=3, name="b4a")
    x = _conv_block(x, 512, kernel=3, name="b4b")
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)           # 3x3

    # Embedding head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    x = layers.Dense(embedding_size, kernel_initializer="he_normal",
                     name="embedding_dense")(x)
    # L2 normalization: embeddings live on the unit hypersphere,
    # so cosine similarity is a simple dot product.
    outputs = layers.UnitNormalization(name="l2_normalization")(x)

    return keras.Model(inputs, outputs, name="facenet")


def build_augmentation():
    """Data augmentation applied only during training (Keras preprocessing
    layers are automatically inactive at inference time)."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.06),           # ~±20 degrees
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.08, 0.08),  # width/height shift
            layers.RandomBrightness(0.25, value_range=(0.0, 1.0)),
        ],
        name="augmentation",
    )


def build_training_model():
    """Wrap augmentation + FaceNet for model.fit; the inner 'facenet'
    sub-model is what gets saved as facenet_model.keras."""
    base = build_facenet()
    inputs = keras.Input(shape=(config.FACE_IMAGE_SIZE, config.FACE_IMAGE_SIZE, 3))
    x = build_augmentation()(inputs)
    outputs = base(x)
    return keras.Model(inputs, outputs, name="facenet_training"), base
