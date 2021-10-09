import tensorflow as tf

from spatial_transform.aff_mnist_data import IMAGE_SIZE, IMAGE_SHAPE, IMAGE_NUM_CHANNELS, get_aff_mnist_data
from spatial_transform.spatial_transforms import AffineTransform
from spatial_transform.st_blocks import SimpleSpatialTransformBlock
from spatial_transform.localization import CoordConvLocalizationLayer
from spatial_transform.interpolation import BilinearInterpolator


# Load data
train_img_data, train_img_label, validation_img_data, validation_img_label, test_img_data, test_img_label = get_aff_mnist_data()

# Init model
image = tf.keras.layers.Input(shape=IMAGE_SHAPE + (IMAGE_NUM_CHANNELS,))
size_after_transform = 30
spatial_transform = AffineTransform()
st_block = SimpleSpatialTransformBlock(
    localization_layer = CoordConvLocalizationLayer(
        spatial_transform_params_cls = spatial_transform.param_type,
        init_scale = size_after_transform / IMAGE_SIZE,
    ),
    spatial_transform = spatial_transform,
    interpolator = BilinearInterpolator(),
    shape_out = (size_after_transform, size_after_transform)
)
x = image
x = st_block(x)
x = tf.keras.layers.Conv2D(32, [7, 7], activation='relu', padding="valid")(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(32, [5, 5], activation='relu', padding="valid")(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(90, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(10, activation=None)(x)
model = tf.keras.models.Model(inputs=image, outputs=x)
model.summary()
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy']
)

# Train model
history = model.fit(
    x = train_img_data,
    y = train_img_label,
    batch_size = 128,
    epochs = 6,
    validation_data = (test_img_data,  test_img_label),
    validation_batch_size = 1024,
)