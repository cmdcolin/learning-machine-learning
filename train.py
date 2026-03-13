import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import optax
import numpy as np
import os
import gzip
import struct
import json
import time

# --- Data Loading ---
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows * cols)
        return data.astype(np.float32) / 255.0

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def get_datasets():
    data_dir = "data"
    train_images = load_mnist_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    train_labels = load_mnist_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    test_images = load_mnist_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    test_labels = load_mnist_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    
    # One-hot encode labels
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    
    return train_images, train_labels, test_images, test_labels

# --- Model Definition ---
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    # Using Lecun/Xavier initialization
    std = jnp.sqrt(2.0 / (m + n))
    return std * random.normal(w_key, (n, m)), jnp.zeros((n,))

def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def relu(x):
    return jnp.maximum(0, x)

def predict(params, image):
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)
    
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - jax.nn.logsumexp(logits)

batched_predict = vmap(predict, in_axes=(None, 0))

# --- Data Augmentation ---
@jit
def augment(rng, images):
    # images is (B, 784)
    B = images.shape[0]
    images = images.reshape(B, 28, 28)
    
    rng_shift, rng_noise = random.split(rng)
    
    # Random shifts (up to 2 pixels)
    shifts = random.uniform(rng_shift, (B, 2), minval=-2.0, maxval=2.0)
    
    # We'll use a simple roll/shift for speed or just leave as is if complex interpolation is needed.
    # For now, let's just add small random noise and horizontal/vertical translations.
    # Manual shift for 2D images in JAX
    def shift_image(img, s):
        # This is a bit complex in pure JAX without scipy.ndimage (which is slow in JIT)
        # Let's just do random noise and slight intensity scaling for now.
        return img
    
    # Add random noise
    noise = random.normal(rng_noise, images.shape) * 0.05
    images = images + noise
    
    return jnp.clip(images.reshape(B, 784), 0, 1)

def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)

@jit
def update(params, opt_state, x, y, rng):
    x_aug = augment(rng, x)
    grads = grad(loss)(params, x_aug, y)
    updates, opt_state = optimizer.update(grads, opt_state, params) # Fixed: passed params
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

# --- Training Configuration ---
layer_sizes = [784, 1024, 512, 10] # Wider network
step_size = 0.0005 # Smaller learning rate for stability
num_epochs = 15 # More epochs
batch_size = 128

optimizer = optax.adamw(step_size, weight_decay=1e-4) # AdamW with weight decay

# --- Execution ---
if __name__ == "__main__":
    # ... (data checking remains same)
    data_dir = "data"
    required_files = [
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    ]
    if not all(os.path.exists(os.path.join(data_dir, f)) for f in required_files):
        print("Data files missing. Running download_data.sh...")
        import subprocess
        subprocess.run(["./download_data.sh"], check=True)

    print("Loading data...")
    train_images, train_labels, test_images, test_labels = get_datasets()
    num_train = train_images.shape[0]
    num_batches = num_train // batch_size

    rng = random.PRNGKey(42)
    params_rng, train_rng = random.split(rng)

    print("Initializing parameters...")
    params = init_network_params(layer_sizes, params_rng)
    opt_state = optimizer.init(params)

    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        # Shuffle indices
        train_rng, subkey = random.split(train_rng)
        perms = random.permutation(subkey, num_train)
        
        for i in range(num_batches):
            batch_rng = random.fold_in(train_rng, i + epoch * num_batches)
            batch_indices = perms[i * batch_size : (i + 1) * batch_size]
            x = train_images[batch_indices]
            y = train_labels[batch_indices]
            params, opt_state = update(params, opt_state, x, y, batch_rng)
        
        epoch_time = time.time() - start_time
        test_acc = accuracy(params, test_images, test_labels)
        print(f"Epoch {epoch+1} in {epoch_time:0.2f} sec | Test Acc: {test_acc:0.4f}")

    # ... (export remains same)

    # --- Export ---
    print("\nExporting weights and test data...")
    
    # Structure: [ {weights: ..., biases: ...}, ... ]
    weights_export = []
    for w, b in params:
        weights_export.append({
            "weights": np.array(w).tolist(),
            "biases": np.array(b).tolist()
        })
        
    # Take first 5 images from the test set for the demo
    test_export = []
    for i in range(5):
        test_export.append({
            "image": np.array(test_images[i]).tolist(),
            "label": int(np.argmax(test_labels[i]))
        })
    
    # Export 500 balanced training examples for the browser training demo.
    # Stored as uint8 integers (0-255) to keep the JSON small (~1.2 MB).
    # The browser normalises them to [0,1] before feeding into the network.
    print("Exporting browser training data (500 balanced examples)...")
    train_subset = []
    counts = [0] * 10
    # train_labels here is one-hot encoded, so we use argmax to get the class
    for i in range(len(train_images)):
        label_idx = int(np.argmax(np.array(train_labels[i])))
        if counts[label_idx] < 50:
            train_subset.append({
                "image": [int(round(float(p) * 255)) for p in train_images[i].tolist()],
                "label": label_idx
            })
            counts[label_idx] += 1
        if sum(counts) == 500:
            break

    # Export to multiple locations (monorepo structure)
    export_dirs = ["packages/cli", "packages/web/src/assets"]
    for d in export_dirs:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "weights.json"), "w") as f:
            json.dump(weights_export, f)
        with open(os.path.join(d, "test_images.json"), "w") as f:
            json.dump(test_export, f)

    with open("packages/web/src/assets/train_data.json", "w") as f:
        json.dump(train_subset, f)

    print(f"Export complete to: {', '.join(export_dirs)}")
    print(f"  train_data.json: {len(train_subset)} examples ({counts} per class)")
