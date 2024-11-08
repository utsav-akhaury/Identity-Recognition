# We use the source code of the InceptionResNetV2 from Keras
# and replace all Conv2D layers by SeperableConv2D layers.
# Some layers were also commented out to reduce the no. of parameters

import tensorflow.compat.v2 as tf

from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from keras.layers import VersionAwareLayers
from keras.utils import data_utils
from keras.utils import layer_utils


BASE_WEIGHT_URL = ('https://storage.googleapis.com/tensorflow/'
                   'keras-applications/inception_resnet_v2/')
layers = None


def InceptionResNetV2(include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=1000,
                      classifier_activation='softmax',
                      **kwargs):
  global layers
  if 'layers' in kwargs:
    layers = kwargs.pop('layers')
  else:
    layers = VersionAwareLayers()
  if kwargs:
    raise ValueError('Unknown argument(s): %s' % (kwargs,))
  if not (weights in {'imagenet', None} or tf.io.gfile.exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                     ' as true, `classes` should be 1000')

  # Determine proper input shape
  input_shape = imagenet_utils.obtain_input_shape(
      input_shape,
      default_size=299,
      min_size=75,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  # Stem block: 35 x 35 x 192
  x = conv2d_bn(img_input, 32, 3, strides=2, padding='valid')
  x = conv2d_bn(x, 32, 3, padding='valid')
  # x = conv2d_bn(x, 64, 3)
  x = layers.MaxPooling2D(3, strides=2)(x)
  x = conv2d_bn(x, 80, 1, padding='valid')
  x = conv2d_bn(x, 192, 3, padding='valid')
  x = layers.MaxPooling2D(3, strides=2)(x)

  # Mixed 5b (Inception-A block): 35 x 35 x 320
  branch_0 = conv2d_bn(x, 96, 1)
  branch_1 = conv2d_bn(x, 48, 1)
  branch_1 = conv2d_bn(branch_1, 64, 5)
  branch_2 = conv2d_bn(x, 64, 1)
  # branch_2 = conv2d_bn(branch_2, 96, 3)
  branch_2 = conv2d_bn(branch_2, 96, 3)
  branch_pool = layers.AveragePooling2D(3, strides=1, padding='same')(x)
  branch_pool = conv2d_bn(branch_pool, 64, 1)
  branches = [branch_0, branch_1, branch_2, branch_pool]
  channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
  x = layers.Concatenate(axis=channel_axis, name='mixed_5b')(branches)

  # 2x block35 (Inception-ResNet-A block): 35 x 35 x 320
  for block_idx in range(1, 3):
    x = inception_resnet_block(
        x, scale=0.17, block_type='block35', block_idx=block_idx)

  # Mixed 6a (Reduction-A block): 17 x 17 x 1088
  branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid')
  branch_1 = conv2d_bn(x, 256, 1)
  # branch_1 = conv2d_bn(branch_1, 256, 3)
  branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')
  branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
  branches = [branch_0, branch_1, branch_pool]
  x = layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)

  # 2x block17 (Inception-ResNet-B block): 17 x 17 x 1088
  for block_idx in range(1, 3):
    x = inception_resnet_block(
        x, scale=0.1, block_type='block17', block_idx=block_idx)

  # Mixed 7a (Reduction-B block): 8 x 8 x 2080
  branch_0 = conv2d_bn(x, 256, 1)
  branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')
  branch_1 = conv2d_bn(x, 256, 1)
  branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')
  branch_2 = conv2d_bn(x, 256, 1)
  # branch_2 = conv2d_bn(branch_2, 288, 3)
  branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')
  branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
  branches = [branch_0, branch_1, branch_2, branch_pool]
  x = layers.Concatenate(axis=channel_axis, name='mixed_7a')(branches)

  # 2x block8 (Inception-ResNet-C block): 8 x 8 x 2080
  for block_idx in range(1, 3):
    x = inception_resnet_block(
        x, scale=0.2, block_type='block8', block_idx=block_idx)
  x = inception_resnet_block(
      x, scale=1., activation=None, block_type='block8', block_idx=10)

  # Final convolution block: 8 x 8 x 1536
  x = conv2d_bn(x, 1536, 1, name='conv_7b')

  if include_top:
    # Classification block
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Dense(classes, activation=classifier_activation,
                     name='predictions')(x)
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D()(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input

  # Create model.
  model = training.Model(inputs, x, name='inception_resnet_v2')

  # Load weights.
  if weights == 'imagenet':
    if include_top:
      fname = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
      weights_path = data_utils.get_file(
          fname,
          BASE_WEIGHT_URL + fname,
          cache_subdir='models',
          file_hash='e693bd0210a403b3192acc6073ad2e96')
    else:
      fname = ('inception_resnet_v2_weights_'
               'tf_dim_ordering_tf_kernels_notop.h5')
      weights_path = data_utils.get_file(
          fname,
          BASE_WEIGHT_URL + fname,
          cache_subdir='models',
          file_hash='d19885ff4a710c122648d3b5c3b684e4')
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  return model


def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):

  x = layers.SeparableConv2D(
      filters,
      kernel_size,
      strides=strides,
      padding=padding,
      use_bias=use_bias,
      name=name)(
          x)
  if not use_bias:
    bn_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    bn_name = None if name is None else name + '_bn'
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
  if activation is not None:
    ac_name = None if name is None else name + '_ac'
    x = layers.Activation(activation, name=ac_name)(x)
  return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):

  if block_type == 'block35':
    branch_0 = conv2d_bn(x, 32, 1)
    branch_1 = conv2d_bn(x, 32, 1)
    branch_1 = conv2d_bn(branch_1, 32, 3)
    branch_2 = conv2d_bn(x, 32, 1)
    # branch_2 = conv2d_bn(branch_2, 48, 3)
    branch_2 = conv2d_bn(branch_2, 64, 3)
    branches = [branch_0, branch_1, branch_2]
  elif block_type == 'block17':
    branch_0 = conv2d_bn(x, 192, 1)
    branch_1 = conv2d_bn(x, 128, 1)
    # branch_1 = conv2d_bn(branch_1, 160, [1, 7])
    branch_1 = conv2d_bn(branch_1, 192, [7, 1])
    branches = [branch_0, branch_1]
  elif block_type == 'block8':
    branch_0 = conv2d_bn(x, 192, 1)
    branch_1 = conv2d_bn(x, 192, 1)
    # branch_1 = conv2d_bn(branch_1, 224, [1, 3])
    branch_1 = conv2d_bn(branch_1, 256, [3, 1])
    branches = [branch_0, branch_1]
  else:
    raise ValueError('Unknown Inception-ResNet block type. '
                     'Expects "block35", "block17" or "block8", '
                     'but got: ' + str(block_type))

  block_name = block_type + '_' + str(block_idx)
  channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
  mixed = layers.Concatenate(
      axis=channel_axis, name=block_name + '_mixed')(
          branches)
  up = conv2d_bn(
      mixed,
      backend.int_shape(x)[channel_axis],
      1,
      activation=None,
      use_bias=True,
      name=block_name + '_conv')

  x = layers.Lambda(
      lambda inputs, scale: inputs[0] + inputs[1] * scale,
      output_shape=backend.int_shape(x)[1:],
      arguments={'scale': scale},
      name=block_name)([x, up])
  if activation is not None:
    x = layers.Activation(activation, name=block_name + '_ac')(x)
  return x