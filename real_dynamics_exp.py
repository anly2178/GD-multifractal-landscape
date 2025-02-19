import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, value_and_grad, random, jvp, flatten_util
from jax.nn import relu
from tqdm import tqdm

# Load the dataset
def load_dataset():
    data = np.loadtxt("data/airfoil_self_noise.dat")
    X, y = data[:, :-1], data[:, -1]
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # Normalize input features
    y = (y - y.mean()) / y.std(axis=0)  # Normalize target
    return X, y

# Split dataset into training and testing sets
def split_dataset(X, y, ntrain):
    train_X, test_X = X[:ntrain], X[ntrain:]
    train_y, test_y = y[:ntrain], y[ntrain:]
    return train_X, train_y, test_X, test_y

# Initialize model parameters
def init_params(rng):
    key1, key2 = random.split(rng)
    return [jnp.sqrt(2 / 5) * random.normal(key1, (5, 16)), jnp.zeros((16,)), jnp.sqrt(2 / 16) * random.normal(key2, (16, 1)), jnp.zeros((1,))] 

# Forward pass with dropout
def forward(params, x, rng, dropout_rate, train=True):
    W1, b1, W2, b2 = params  # Unpack weights and biases
    key1, key2 = random.split(rng)
    h = relu(jnp.dot(x, W1) + b1)
    if train:
        if dropout_rate > 0:
            mask = random.bernoulli(key1, 1 - dropout_rate, h.shape)
            h = h * mask / (1 - dropout_rate)
    out = jnp.dot(h, W2) + b2
    return out

# Loss function (MSE)
def mse_loss(params, x, y, rng, dropout_rate, train=True):
    preds = forward(params, x, rng, dropout_rate, train).reshape(y.shape)
    return jnp.mean((preds - y) ** 2)

# Hessian-vector product
def hvp(f, primals, tangents, *args):
    return jvp(grad(lambda p: f(p, *args)), (primals,), (tangents,))[1]

def norm(params):
    return jnp.sqrt(jnp.sum(jnp.array([jnp.sum(p**2) for p in params]))) ###

# Compute the maximum Hessian eigenvalue by power iteration method
def max_hessian_eigenvalue(params, x, y, rng, dropout_rate, steps=20, error_threshold=1e-4):
    def random_init_fn(rng, param):
        rng, subkey = random.split(rng)
        return random.normal(subkey, shape=param.shape)
    prev_lambda = 0.0
    vec = jax.tree_util.tree_map(lambda p: random_init_fn(rng, p), params) # Random vector
    for i in range(steps):
        new_vec = hvp(mse_loss, params, vec, x, y, rng, dropout_rate)
        if norm(new_vec) == 0.0:
            return 0.0, new_vec
        lambda_estimate = jnp.sum(jnp.array([jnp.sum(v * nv) for v, nv in zip(vec, new_vec)])) 
        diff = lambda_estimate - prev_lambda
        vec = [nv / norm(new_vec) for nv in new_vec] 
        if lambda_estimate == 0.0: # low-rank
            error = 1.0
        else:
            error = jnp.abs(diff / lambda_estimate)
        if error < error_threshold:
            break
        prev_lambda = lambda_estimate
    return lambda_estimate, vec

# Training loop
def train(params, train_X, train_y, test_X, test_y, epochs, lr, dropout_rate, hess_times):
    rng = random.PRNGKey(42)

    train_losses = []
    test_losses = []
    max_eigs = []
    trajectory = []
    for epoch in tqdm(range(epochs)):
        rng, subkey = random.split(rng)

        train_loss, grads = value_and_grad(mse_loss)(params, train_X, train_y, subkey, dropout_rate, train=True)
        params = jax.tree_util.tree_map(lambda t, g: t - lr*g, params, grads)

        # Compute test loss
        test_loss = mse_loss(params, test_X, test_y, subkey, dropout_rate, train=False)

        # Compute Hessian max eigenvalue
        if epoch in hess_times:
            max_eig, _ = max_hessian_eigenvalue(params, train_X, train_y, subkey, dropout_rate)
            max_eigs.append(max_eig)

        # Save
        params_vec, unravel_fn = flatten_util.ravel_pytree(params)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        trajectory.append(params_vec)
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.5f}, Test Loss={test_loss:.5f}, Max Hessian Eigen={max_eig:.5f}")

    return train_losses, test_losses, max_eigs, trajectory

def create_directory(lr, dropout, epochs):
    directory = f"results/lr={lr}_dropout={dropout}_epochs={epochs}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# Main function
def main():
    parser = argparse.ArgumentParser(description="Train a JAX MLP on Airfoil Self Noise")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=1000000, help="Number of epochs")
    parser.add_argument("--ntrain", type=int, default=1503, help="Size of training set. Max is 1503.")
    parser.add_argument("--hess_every", type=int, default=10, help="Evaluate maximum Hessian eigenvalue every...")
    parser.add_argument("--hess_term", type=int, default=1000, help="Terminate Hessian calculation after...")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    args = parser.parse_args()

    X, y = load_dataset()
    train_X, train_y, test_X, test_y = split_dataset(X, y, args.ntrain)

    rng = random.PRNGKey(args.seed)
    params = init_params(rng)

    hess_times = jnp.arange(0, args.hess_term, args.hess_every)
    train_losses, test_losses, max_eigs, trajectory = train(params, train_X, train_y, test_X, test_y, args.epochs, args.lr, args.dropout, hess_times)

    directory = create_directory(args.lr, args.dropout, args.epochs)
    np.savez(directory + "/learning_curves.npz", train_losses=train_losses, test_losses=test_losses)
    np.save(directory + "/max_eigs.npy", max_eigs)
    np.save(directory + "/trajectory.npy", trajectory)

if __name__ == "__main__":
    main()