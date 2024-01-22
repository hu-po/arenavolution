import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
import tensorflow_datasets as tfds
import jax
import numpy as np

num_epochs = 10
batch_size = 32
learning_rate = 0.001
num_classes = 1000

def preprocess_image(image, label):
    # Resize the image to 224x224 and normalize pixel values
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def load_dataset(split, batch_size):
    # Load the specified split of the dataset
    dataset = tfds.load('imagenet2012', split=split, as_supervised=True)

    # Preprocess the dataset
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Batch and prefetch the dataset
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def to_jax_dataset(tf_dataset):
    # Convert TensorFlow dataset to a JAX-compatible format (NumPy arrays)
    images, labels = [], []
    for img, lbl in tf_dataset:
        images.append(img.numpy())
        labels.append(lbl.numpy())
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    return {'images': images, 'labels': labels}

# Load datasets
batch_size = 32  # Adjust batch size according to your system's capabilities

train_tf_dataset = load_dataset('train', batch_size)
val_tf_dataset = load_dataset('validation', batch_size)
test_tf_dataset = load_dataset('test', batch_size)

# Convert to JAX datasets
train_dataset = to_jax_dataset(train_tf_dataset)
val_dataset = to_jax_dataset(val_tf_dataset)
test_dataset = to_jax_dataset(test_tf_dataset)


class ConvNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.MaxPool(window_shape=(2, 2), strides=(2, 2))(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.MaxPool(window_shape=(2, 2), strides=(2, 2))(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=num_classes)(x)
        return x

def cross_entropy_loss(logits, labels):
    return optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, num_classes)).mean()

def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = ConvNet().apply({'params': params}, batch['images'])
        loss = cross_entropy_loss(logits, batch['labels'])
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(params, batch):
    logits = ConvNet().apply({'params': params}, batch['images'])
    return compute_metrics(logits, batch['labels'])

model = ConvNet()
params = model.init(jax.random.PRNGKey(0), jnp.ones([1, 224, 224, 3]))['params']
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.adam(learning_rate)
)

for epoch in range(num_epochs):
    for batch in train_dataset:
        state, loss = train_step(state, batch)
        print(f'Epoch {epoch}, Loss: {loss}')

    val_metrics = []
    for batch in val_dataset:
        metrics = eval_step(state.params, batch)
        val_metrics.append(metrics)
    val_metrics = jax.device_get(jax.tree_multimap(lambda *args: np.mean(args), *val_metrics))
    print(f'Validation Epoch {epoch}, Metrics: {val_metrics}')

test_metrics = []
for batch in test_dataset:
    metrics = eval_step(state.params, batch)
    test_metrics.append(metrics)
test_metrics = jax.device_get(jax.tree_multimap(lambda *args: np.mean(args), *test_metrics))
print(f'Test Metrics: {test_metrics}')