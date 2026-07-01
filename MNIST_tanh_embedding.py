#!/usr/bin/env python3
"""
Scalar embedding of MNIST neural-network training trajectories.

This script implements the MNIST baseline used to study scalar embeddings of
neural-network training dynamics. It trains a fully connected MLP with tanh
activation using deterministic full-batch gradient descent, generates nearby
perturbed training trajectories, computes distances between trajectories in
parameter space, constructs one-dimensional Classical MDS embeddings, and 
compute basic post-processing observables from the embedded trajectories.

Main idea
---------
For each learning rate and random initial condition, the script:

1. Trains a reference network and records its parameter vector w(t).
2. Builds several perturbed copies of the same initial network.
3. Trains each perturbed copy under the same optimization protocol.
4. Computes distance trajectories between reference and perturbed networks.
5. Builds a pairwise distance matrix between all trajectory snapshots.
6. Applies Classical MDS to obtain a scalar embedding of the training dynamics.
7. Optionally post-processes the saved embeddings to compute decorrelation
   times, asymptotic embedded states, and nearest-neighbor spacings.

Notes
-----
- Training is full-batch and deterministic: no SGD mini-batch noise is used.
- Perturbations are applied only to weights, not biases.
- The default entry point expects a SLURM array job. The environment variable
  SLURM_ARRAY_TASK_ID selects the learning rate from ``learning_rate_array``.
- Outputs are written as compressed ``.npz`` files under ``results/`` or
  ``results_shards/`` depending on the selected main routine.
- Extended post-processing steps such as Lyapunov exponent fitting, statistical 
  analysis across learning rates, and final figure generation are not included.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector
from torchvision import datasets, transforms
from scipy import signal
from scipy.stats import pearsonr
from scipy.signal import savgol_filter


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------

# MNIST images are converted to tensors without normalization
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("../data", train=False, download=True, transform=transform)

# Full-batch loaders: each epoch consists of a single gradient-descent update.
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=len(train_dataset),
    shuffle=False,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=len(test_dataset),
    shuffle=False,
)

# Keep the full test set in memory for fast evaluation after each epoch.
x_train, y_train = next(iter(train_loader))
x_test, y_test = next(iter(test_loader))


# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------

class MNISTNet(nn.Module):
    """
    One-hidden-layer fully connected MLP for MNIST.

    Architecture:
        784 input pixels -> 64 tanh hidden units -> 10 output logits

    Parameters are initialized from a normal distribution with zero biases,
    unless an existing ``initial_state`` is provided.
    """

    def __init__(self, input_dim=784, hidden_units=64, output_dim=10, initial_state=None):
        super(MNISTNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_units).double()
        self.fc2 = nn.Linear(hidden_units, output_dim).double()

        if initial_state is not None:
            self.load_state_dict(initial_state)
        else:
            self.init_weights()

    def init_weights(self):
        """Initialize Linear layers with N(0, 1) weights and zero biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0, std=1)
                module.bias.data.fill_(0)

    def forward(self, x):
        """Forward pass for flattened MNIST images."""
        x = x.view(-1, 784)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------------------------------------------------------
# Distance and embedding utilities
# -----------------------------------------------------------------------------

def network_distance_torch(arr1, arr2):
    """Root-mean-square distance between two flattened network snapshots."""
    return torch.sqrt(torch.sum((arr1 - arr2) ** 2) / arr1.numel()).item()


def distance_matrix_labelled_TN_torch(TN):
    """
    Compute the pairwise distance matrix between trajectory snapshots.

    Parameters
    ----------
    TN : torch.Tensor
        Tensor whose first dimension indexes snapshots and whose second
        dimension contains flattened model parameters.

    Returns
    -------
    numpy.ndarray
        Symmetric pairwise distance matrix.
    """
    dimension = len(TN)
    distance_matrix = np.zeros([dimension, dimension])

    for i in range(dimension):
        for j in range(i + 1, dimension):
            distance_matrix[i, j] = network_distance_torch(TN[i], TN[j])
            distance_matrix[j, i] = distance_matrix[i, j]

    return distance_matrix


def my_own_MDS(D, m):
    """
    Classical Multidimensional Scaling (MDS).

    Parameters
    ----------
    D : numpy.ndarray
        Pairwise distance matrix.
    m : int
        Number of embedding dimensions.

    Returns
    -------
    numpy.ndarray
        Coordinate matrix with ``m`` embedding dimensions.
    """
    # Square distances.
    D_squared = np.square(D)

    # Double-centering step.
    n = len(D)
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_squared @ H

    # Eigen-decomposition of the centered Gram matrix.
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    sorted_indices = np.argsort(eigenvalues)[::-1]

    largest_eigenvalues = eigenvalues[sorted_indices][:m]
    largest_eigenvectors = eigenvectors[:, sorted_indices][:, :m]

    # Build the coordinate matrix.
    Lambda_m = np.diag(np.sqrt(largest_eigenvalues))
    X = np.dot(largest_eigenvectors, Lambda_m)

    return X


def get_flat_params(model):
    """Return a detached flattened copy of all model parameters."""
    return parameters_to_vector(model.parameters()).clone()


def distance_trajectories_flat(flat_params_orig, flat_params_pert):
    """
    Compute the L1 distance trajectory between two parameter trajectories.

    Both input lists should have length ``num_epochs + 1`` because the initial
    state is stored before the first optimization step.
    """
    distances = []

    for vec_orig, vec_pert in zip(flat_params_orig, flat_params_pert):
        distances.append((vec_orig - vec_pert).abs().sum().item())

    return distances


# -----------------------------------------------------------------------------
# Training routines
# -----------------------------------------------------------------------------

def initialize_network_and_flatten(num_epochs, learning_rate, initial_weights=None):
    """
    Train a reference network and store its flattened parameter trajectory.

    Parameters
    ----------
    num_epochs : int
        Number of full-batch gradient-descent steps.
    learning_rate : float
        Gradient-descent learning rate.
    initial_weights : dict or None, optional
        Optional state dictionary used to initialize the network. If ``None``,
        the network is initialized randomly.

    Returns
    -------
    initial_state : dict
        Initial network state before training.
    losses : numpy.ndarray
        Training loss at each epoch.
    flat_params : list[torch.Tensor]
        Flattened parameter vectors, including the initial state.
    test_accs : numpy.ndarray
        Test accuracy at each epoch.
    """
    input_dim = 784
    hidden_units = 64
    output_dim = 10

    net = MNISTNet(input_dim, hidden_units, output_dim, initial_weights)

    # Use float tensors in the actual forward/backward pass.
    for param in net.parameters():
        param.data = param.data.float()

    initial_state = {k: v.clone() for k, v in net.state_dict().items()}

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0)

    losses = np.zeros(num_epochs)
    test_accs = np.zeros(num_epochs)

    # Store the initial condition before any optimization step.
    flat_params = [get_flat_params(net)]

    for epoch in range(num_epochs):
        net.train()

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs.float())
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        losses[epoch] = loss.item()
        flat_params.append(get_flat_params(net))

        net.eval()
        with torch.no_grad():
            logits = net(x_test.float())
            preds = logits.argmax(dim=1)
            test_acc = (preds == y_test).float().mean().item()

        test_accs[epoch] = test_acc

    return initial_state, losses, flat_params, test_accs


def pert_initial_conditions_flat(num_epochs, learning_rate, initial_state, eps, num_pert):
    """
    Train perturbed copies of a reference initial condition.

    Each perturbed network starts from ``initial_state`` plus a uniform random
    perturbation in ``[-eps, eps]`` applied to weights only. Biases are kept
    unchanged.
    """
    input_dim = 784
    hidden_units = 64
    output_dim = 10

    pert_losses = []
    pert_flat_params = []
    pert_acc = []

    for j in range(num_pert):
        net_pert = MNISTNet(input_dim, hidden_units, output_dim, initial_state)

        for param in net_pert.parameters():
            param.data = param.data.float()

        with torch.no_grad():
            for name, param in net_pert.named_parameters():
                if name.endswith("bias"):
                    continue
                param.add_(torch.rand_like(param) * 2 * eps - eps)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net_pert.parameters(), lr=learning_rate, momentum=0)

        losses = np.zeros(num_epochs)
        test_accs = np.zeros(num_epochs)
        flat_params = [get_flat_params(net_pert)]

        for epoch in range(num_epochs):
            net_pert.train()

            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = net_pert(inputs.float())
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            losses[epoch] = loss.item()
            flat_params.append(get_flat_params(net_pert))

            net_pert.eval()
            with torch.no_grad():
                logits = net_pert(x_test.float())
                preds = logits.argmax(dim=1)
                test_acc = (preds == y_test).float().mean().item()

            test_accs[epoch] = test_acc

        pert_losses.append(losses)
        pert_flat_params.append(flat_params)
        pert_acc.append(test_accs)

    return pert_losses, pert_flat_params, pert_acc


# -----------------------------------------------------------------------------
# Experiment routines
# -----------------------------------------------------------------------------

def lyapunov_embedding(learning_rate):
    """
    Run the standard embedding experiment for one learning rate.

    This routine uses several independent random initial conditions. For each
    one, it trains a reference trajectory and multiple perturbed trajectories,
    then stores distances, losses, accuracies, and scalar MDS coordinates.
    """
    n_init = 50
    num_epochs = 200
    epsilon = np.array([1e-8])
    num_pert = 5

    # Probe the number of parameters and trajectory length once.
    initial_state, _, flat_traj_orig_probe, _ = initialize_network_and_flatten(
        num_epochs=0,
        learning_rate=learning_rate,
    )
    probe = flat_traj_orig_probe[0].view(-1)
    P = probe.numel()
    del initial_state, flat_traj_orig_probe, probe

    L = num_epochs + 1                      # Include the initial state.
    N = (num_pert + 1) * L                  # Reference + perturbations.

    mds_all = np.zeros((n_init, N), dtype=np.float32)

    loss_orig_all = np.zeros((n_init, num_epochs), dtype=np.float16)
    loss_pert_all = np.zeros((n_init, num_pert, num_epochs), dtype=np.float16)

    acc_orig_all = np.zeros((n_init, num_epochs), dtype=np.float16)
    acc_pert_all = np.zeros((n_init, num_pert, num_epochs), dtype=np.float16)

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/MNIST_embeddings_tanh_lr_{learning_rate}.npz"

    d = np.zeros((len(epsilon), n_init, num_pert, num_epochs + 1), dtype=np.float32)

    tic = time.time()
    print(f"Starting lr = {learning_rate} ...")

    for eps_idx, eps in enumerate(epsilon):
        for i in range(n_init):
            ti = time.time()

            # Reference trajectory.
            initial_state, loss_orig, flat_traj_orig, acc_orig = initialize_network_and_flatten(
                num_epochs,
                learning_rate,
            )

            loss_orig_all[i] = np.asarray(loss_orig, dtype=np.float16)
            acc_orig_all[i] = np.asarray(acc_orig, dtype=np.float16)

            # Perturbed trajectories starting from the same initial condition.
            pert_loss, pert_flat_trajs, pert_acc = pert_initial_conditions_flat(
                num_epochs,
                learning_rate,
                initial_state,
                eps,
                num_pert,
            )

            loss_pert_all[i] = np.asarray(pert_loss, dtype=np.float16)
            acc_pert_all[i] = np.asarray(pert_acc, dtype=np.float16)

            # L1 divergence trajectories d(t) between reference and perturbations.
            distances = [
                distance_trajectories_flat(flat_traj_orig, pert_traj)
                for pert_traj in pert_flat_trajs
            ]
            d[eps_idx, i, :, :] = np.asarray(distances, dtype=np.float32)

            # Build the full set of snapshots used for scalar MDS.
            orig_mat = torch.stack([t.view(-1) for t in flat_traj_orig], dim=0)       # (L, P)
            pert_mats = [
                torch.stack([t.view(-1) for t in seq], dim=0)                         # (L, P)
                for seq in pert_flat_trajs
            ]

            concat_mat = torch.cat([orig_mat] + pert_mats, dim=0)                     # (N, P)
            distance_matrix = distance_matrix_labelled_TN_torch(concat_mat)

            del concat_mat, orig_mat, pert_mats

            num_components = 1
            classic_mds = np.asarray(np.squeeze(my_own_MDS(distance_matrix, num_components)))
            mds_all[i] = classic_mds

            to = time.time()
            print(f"Finished C.I. {i} after {to - ti:.2f}s")

    toc = time.time()
    print(f"Finished job in {toc - tic:.2f}s")

    np.savez_compressed(
        out_path,
        classic_mds=mds_all,
        loss_orig_all=loss_orig_all,
        loss_pert_all=loss_pert_all,
        distances=d,
        accuracy_original=acc_orig_all,
        accuracy_perturbed=acc_pert_all,
    )


def lyapunov_embedding_1(learning_rate, initial_state):
    """
    Run a fixed-initial-condition learning-rate sweep.

    This version is useful when all simulations should start from the same
    initial condition, for example when scanning eta more finely.
    """
    n_init = 1
    num_epochs = 200
    epsilon = np.array([1e-8])
    num_pert = 25

    _, _, flat_traj_orig_probe, _ = initialize_network_and_flatten(
        num_epochs=0,
        learning_rate=learning_rate,
        initial_weights=initial_state,
    )
    probe = flat_traj_orig_probe[0].view(-1)
    P = probe.numel()
    del flat_traj_orig_probe, probe

    L = num_epochs + 1
    N = (num_pert + 1) * L

    mds_all = np.zeros((n_init, N), dtype=np.float32)

    loss_orig_all = np.zeros((n_init, num_epochs), dtype=np.float16)
    loss_pert_all = np.zeros((n_init, num_pert, num_epochs), dtype=np.float16)

    acc_orig_all = np.zeros((n_init, num_epochs), dtype=np.float16)
    acc_pert_all = np.zeros((n_init, num_pert, num_epochs), dtype=np.float16)

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/barrido_eta_lr_{learning_rate}.npz"

    d = np.zeros((len(epsilon), n_init, num_pert, num_epochs + 1), dtype=np.float32)

    tic = time.time()
    print(f"Starting lr = {learning_rate} ...")

    for eps_idx, eps in enumerate(epsilon):
        for i in range(n_init):
            ti = time.time()

            _, loss_orig, flat_traj_orig, acc_orig = initialize_network_and_flatten(
                num_epochs,
                learning_rate,
                initial_weights=initial_state,
            )

            loss_orig_all[i] = np.asarray(loss_orig, dtype=np.float16)
            acc_orig_all[i] = np.asarray(acc_orig, dtype=np.float16)

            pert_loss, pert_flat_trajs, pert_acc = pert_initial_conditions_flat(
                num_epochs,
                learning_rate,
                initial_state,
                eps,
                num_pert,
            )

            loss_pert_all[i] = np.asarray(pert_loss, dtype=np.float16)
            acc_pert_all[i] = np.asarray(pert_acc, dtype=np.float16)

            distances = [
                distance_trajectories_flat(flat_traj_orig, pert_traj)
                for pert_traj in pert_flat_trajs
            ]
            d[eps_idx, i, :, :] = np.asarray(distances, dtype=np.float32)

            orig_mat = torch.stack([t.view(-1) for t in flat_traj_orig], dim=0)
            pert_mats = [
                torch.stack([t.view(-1) for t in seq], dim=0)
                for seq in pert_flat_trajs
            ]

            concat_mat = torch.cat([orig_mat] + pert_mats, dim=0)
            distance_matrix = distance_matrix_labelled_TN_torch(concat_mat)

            del concat_mat, orig_mat, pert_mats

            num_components = 1
            classic_mds = np.asarray(np.squeeze(my_own_MDS(distance_matrix, num_components)))
            mds_all[i] = classic_mds

            to = time.time()
            print(f"Finished C.I. {i} after {to - ti:.2f}s")

    toc = time.time()
    print(f"Finished job in {toc - tic:.2f}s")

    np.savez_compressed(
        out_path,
        classic_mds=mds_all,
        loss_orig_all=loss_orig_all,
        loss_pert_all=loss_pert_all,
        distances=d,
        accuracy_original=acc_orig_all,
        accuracy_perturbed=acc_pert_all,
    )


def p_delta_embedding(learning_rate, init_idx, initial_state, out_dir="results_shards"):
    """
    Run one initial-condition shard for a fixed learning rate.

    This routine is intended for large SLURM job arrays. Each job computes only
    one initial condition and writes an independent ``.npz`` shard, which makes
    the experiment easier to resume and reduces the risk of losing results.
    """
    num_epochs = 200
    epsilon = np.array([1e-8])
    num_pert = 25

    L = num_epochs + 1
    N = (num_pert + 1) * L

    mds_i = np.zeros((N,), dtype=np.float32)
    acc_orig_i = np.zeros((num_epochs,), dtype=np.float16)
    acc_pert_i = np.zeros((num_pert, num_epochs), dtype=np.float16)

    print(f"[LR={learning_rate}, CI={init_idx}] Starting...")

    for eps_idx, eps in enumerate(epsilon):
        ti = time.time()

        _, _, flat_traj_orig, acc_orig = initialize_network_and_flatten(
            num_epochs,
            learning_rate,
            initial_weights=initial_state,
        )
        acc_orig_i[:] = np.asarray(acc_orig, dtype=np.float16)

        _, pert_flat_trajs, pert_acc = pert_initial_conditions_flat(
            num_epochs,
            learning_rate,
            initial_state,
            eps,
            num_pert,
        )
        acc_pert_i[:, :] = np.asarray(pert_acc, dtype=np.float16)

        orig_mat = torch.stack([t.view(-1) for t in flat_traj_orig], dim=0)
        pert_mats = [
            torch.stack([t.view(-1) for t in seq], dim=0)
            for seq in pert_flat_trajs
        ]

        concat_mat = torch.cat([orig_mat] + pert_mats, dim=0)
        distance_matrix = distance_matrix_labelled_TN_torch(concat_mat)

        del concat_mat, orig_mat, pert_mats

        num_components = 1
        classic_mds = np.asarray(np.squeeze(my_own_MDS(distance_matrix, num_components)))
        mds_i[:] = classic_mds.astype(np.float32)

        to = time.time()
        print(f"[lr={learning_rate}] Finished CI={init_idx} in {to - ti:.2f}s")

    lr_tag = f"{learning_rate}".replace(".", "p")
    shard_dir = os.path.join(out_dir, f"lr_{lr_tag}")
    os.makedirs(shard_dir, exist_ok=True)

    shard_path = os.path.join(shard_dir, f"init_{init_idx:03d}.npz")
    np.savez_compressed(
        shard_path,
        classic_mds=mds_i[None, :],
        accuracy_original=acc_orig_i[None, :],
        accuracy_perturbed=acc_pert_i[None, :, :],
    )

    print("Saved:", shard_path)

# -----------------------------------------------------------------------------
# Post-processing utilities
# -----------------------------------------------------------------------------

def load_embedding_results(path):
    """
    Load a saved embedding experiment from a compressed ``.npz`` file.

    This is the starting point for the analysis stage, after the trajectories
    have already been generated with ``lyapunov_embedding``,
    ``lyapunov_embedding_1``, or ``p_delta_embedding``.
    """
    return dict(np.load(path))


def infer_num_trajectories_from_accuracy(accuracy_perturbed):
    """
    Infer the number of trajectories stored per initial condition.

    The saved files contain one reference trajectory plus ``num_pert`` perturbed
    trajectories. Therefore:

        num_traj = num_pert + 1

    Parameters
    ----------
    accuracy_perturbed : numpy.ndarray
        Array with shape ``(n_init, num_pert, num_epochs)``.

    Returns
    -------
    int
        Total number of trajectories per initial condition.
    """
    num_pert = accuracy_perturbed.shape[1]
    return num_pert + 1


def reshape_embedding(classic_mds, num_traj, num_epochs=None):
    """
    Reshape the scalar MDS output into trajectory form.

    The raw saved array ``classic_mds`` has shape:

        (n_init, num_traj * L)

    where ``L = num_epochs + 1`` because the initial state is included. This
    function reshapes it into:

        (n_init, num_traj, L)

    with trajectory index 0 corresponding to the reference trajectory and
    indices 1..num_pert corresponding to perturbed replicas.

    Parameters
    ----------
    classic_mds : numpy.ndarray
        Raw scalar embedding saved in the ``.npz`` file.
    num_traj : int
        Number of trajectories per initial condition. This is usually
        ``num_pert + 1``.
    num_epochs : int or None, optional
        Number of training epochs. If ``None``, it is inferred from the length
        of the saved embedding.

    Returns
    -------
    numpy.ndarray
        Reshaped scalar embedding with shape ``(n_init, num_traj, L)``.
    """
    classic_mds = np.asarray(classic_mds)

    if classic_mds.ndim == 1:
        classic_mds = classic_mds[None, :]

    n_init, total_length = classic_mds.shape

    if total_length % num_traj != 0:
        raise ValueError(
            "The embedding length is not divisible by num_traj. "
            f"Got total_length={total_length}, num_traj={num_traj}."
        )

    L = total_length // num_traj

    if num_epochs is not None and L != num_epochs + 1:
        raise ValueError(
            "Inconsistent number of epochs. "
            f"Embedding implies L={L}, but num_epochs+1={num_epochs + 1}."
        )

    return classic_mds.reshape(n_init, num_traj, L)


def get_reference_and_perturbed_embedding(embedding_reshaped):
    """
    Split a reshaped embedding into reference and perturbed trajectories.

    Parameters
    ----------
    embedding_reshaped : numpy.ndarray
        Array with shape ``(n_init, num_traj, L)``.

    Returns
    -------
    z_ref : numpy.ndarray
        Reference trajectories with shape ``(n_init, L)``.
    z_pert : numpy.ndarray
        Perturbed trajectories with shape ``(n_init, num_pert, L)``.
    """
    z_ref = embedding_reshaped[:, 0, :]
    z_pert = embedding_reshaped[:, 1:, :]
    return z_ref, z_pert


def embedding_distance_to_reference(embedding_reshaped):
    """
    Compute embedded distances between reference and perturbed trajectories.

    The distance is simply

        d_z(t) = |z_pert(t) - z_ref(t)|

    because the embedding is one-dimensional.

    Parameters
    ----------
    embedding_reshaped : numpy.ndarray
        Array with shape ``(n_init, num_traj, L)``.

    Returns
    -------
    numpy.ndarray
        Distances with shape ``(n_init, num_pert, L)``.
    """
    z_ref, z_pert = get_reference_and_perturbed_embedding(embedding_reshaped)
    return np.abs(z_pert - z_ref[:, None, :])


def tau_decor_from_derivative(
    d,
    t=None,
    slope_fraction=0.15,
    min_epoch=10,
    persistence=8,
    eps=1e-12,
):
    """
    Estimate embedding-based decorrelation time from one embedded distance.

    Given

        d(t) = |z_pert(t) - z_ref(t)|,

    the method computes the numerical derivative of log(d(t)), i.e. the slope
    in semi-log scale, and defines tau_dec as the first epoch at which this
    slope drops below a fraction of its maximum value and remains low for a
    given number of consecutive epochs.

    Parameters
    ----------
    d : array-like
        Embedded distance trajectory.
    t : array-like or None
        Epoch array. If None, t = np.arange(len(d)).
    slope_fraction : float
        Fraction of the maximum slope used as threshold.
    min_epoch : int
        Ignore epochs before this value.
    persistence : int
        Number of consecutive epochs for which the slope must remain below the
        threshold.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    tau : float
        Estimated decorrelation time. Returns np.nan if no persistent drop is
        detected.
    slope : numpy.ndarray
        Numerical derivative of log(d(t)).
    logd : numpy.ndarray
        Log-distance trajectory.
    threshold : float
        Threshold used for detection.
    """
    d = np.asarray(d, dtype=float)

    if t is None:
        t = np.arange(len(d), dtype=float)
    else:
        t = np.asarray(t, dtype=float)

    if len(d) == 0:
        return np.nan, np.array([]), np.array([]), np.nan

    if len(d) != len(t):
        raise ValueError("d and t must have the same length.")

    logd = np.log(d + eps)

    # Slope in semi-log scale.
    slope = np.gradient(logd, t)

    valid = t >= min_epoch

    if not np.any(valid):
        return np.nan, slope, logd, np.nan

    max_slope = np.max(slope[valid])

    if not np.isfinite(max_slope) or max_slope <= 0:
        return np.nan, slope, logd, np.nan

    threshold = slope_fraction * max_slope
    start = np.where(valid)[0][0]

    for i in range(start, len(slope) - persistence + 1):
        if np.all(slope[i:i + persistence] < threshold):
            return float(t[i]), slope, logd, threshold

    return np.nan, slope, logd, threshold


def tau_dec_from_embedding(
    embedding_reshaped,
    slope_fraction=0.15,
    min_epoch=10,
    persistence=8,
    eps=1e-12,
):
    """
    Compute one embedding-based decorrelation time per initial condition.

    Parameters
    ----------
    embedding_reshaped : numpy.ndarray
        Scalar embedding with shape (n_init, num_traj, L). Trajectory 0 is the
        reference and trajectories 1: are perturbed replicas.

    Returns
    -------
    tau_decorr_CI_emb : numpy.ndarray
        One tau_dec value per initial condition, obtained by averaging over
        perturbations.
    """
    embedding_reshaped = np.asarray(embedding_reshaped, dtype=float)

    if embedding_reshaped.ndim != 3:
        raise ValueError("embedding_reshaped must have shape (n_init, num_traj, L).")

    n_init, num_traj, _ = embedding_reshaped.shape
    num_pert = num_traj - 1

    tau_pairs = np.full((n_init, num_pert), np.nan, dtype=float)

    for i in range(n_init):
        z_ref = embedding_reshaped[i, 0]

        for j in range(num_pert):
            z_pert = embedding_reshaped[i, j + 1]
            d = np.abs(z_pert - z_ref)

            tau, *_ = tau_decor_from_derivative(
                d,
                slope_fraction=slope_fraction,
                min_epoch=min_epoch,
                persistence=persistence,
                eps=eps,
            )

            tau_pairs[i, j] = tau

    tau_decorr_CI_emb = np.nanmean(tau_pairs, axis=1)

    return tau_decorr_CI_emb


def compute_tau_corr_null_with_p(
    acc1,
    acc2,
    window=20,
    step=1,
    n_null=500,
    ci=95,
    K=1,
    p_threshold=0.05,
    r_threshold=0.5,
    use_detrend=True,
    null_method="circular_shift",
):
    """
    Estimate the accuracy-based decorrelation time between two trajectories.

    The method computes the Pearson correlation between two accuracy time series
    inside sliding windows. For each window, the observed correlation is compared
    against a null model generated by shifting or permuting one trajectory.

    A window is considered decorrelated when:

        1. The observed Pearson correlation lies inside the null-model band.
        2. The Pearson p-value is larger than ``p_threshold``.
        3. The absolute correlation is smaller than ``r_threshold``.

    The decorrelation time is defined as the center of the first window for which
    the condition holds for ``K`` consecutive windows.

    Parameters
    ----------
    acc1, acc2 : array-like
        Accuracy trajectories. They do not need to have exactly the same length;
        the shortest common length is used.
    window : int
        Sliding-window length.
    step : int
        Sliding-window step.
    n_null : int
        Number of null-model samples per window.
    ci : float
        Confidence interval percentage for the null band.
    K : int
        Number of consecutive windows required to define decorrelation.
    p_threshold : float
        Window is accepted when ``p_value > p_threshold``.
    r_threshold : float
        Window is accepted when ``abs(r) < r_threshold``.
    use_detrend : bool
        If True, linearly detrend each window before computing correlations.
    null_method : {"circular_shift", "permute"}
        Null-model method. ``"circular_shift"`` preserves the temporal structure
        inside the window and is the recommended option.

    Returns
    -------
    tau : float
        Accuracy-based decorrelation time, defined as the center of the first
        persistent decorrelated window. Returns ``np.nan`` if no such window is
        found.
    centers : numpy.ndarray
        Centers of the sliding windows.
    r_real : numpy.ndarray
        Observed Pearson correlations.
    p_vals : numpy.ndarray
        Pearson p-values.
    low : numpy.ndarray
        Lower bound of the null-model confidence interval.
    high : numpy.ndarray
        Upper bound of the null-model confidence interval.
    condition : numpy.ndarray
        Boolean array indicating which windows satisfy the decorrelation
        criterion.
    """
    acc1 = np.asarray(acc1)
    acc2 = np.asarray(acc2)

    T = min(len(acc1), len(acc2))
    acc1 = acc1[:T]
    acc2 = acc2[:T]

    centers = []
    r_real = []
    p_vals = []
    low_list = []
    high_list = []

    alpha_low = (100 - ci) / 2
    alpha_high = 100 - alpha_low

    for start in range(0, T - window + 1, step):
        end = start + window

        x_raw = acc1[start:end]
        y_raw = acc2[start:end]

        if use_detrend:
            x = signal.detrend(x_raw)
            y = signal.detrend(y_raw)
        else:
            x, y = x_raw, y_raw

        r, p = pearsonr(x, y)

        r_real.append(r)
        p_vals.append(p)
        centers.append(start + window // 2)

        null_r = np.empty(n_null, dtype=float)

        if null_method == "circular_shift":
            # Circular shifts preserve the temporal structure within the window.
            for k in range(n_null):
                shift = np.random.randint(1, window)
                y_null_raw = np.roll(y_raw, shift)

                if use_detrend:
                    y_null = signal.detrend(y_null_raw)
                else:
                    y_null = y_null_raw

                r0, _ = pearsonr(x, y_null)
                null_r[k] = r0

        elif null_method == "permute":
            # Less appropriate for time series, but useful as a comparison.
            for k in range(n_null):
                y_null_raw = np.random.permutation(y_raw)

                if use_detrend:
                    y_null = signal.detrend(y_null_raw)
                else:
                    y_null = y_null_raw

                r0, _ = pearsonr(x, y_null)
                null_r[k] = r0

        else:
            raise ValueError("null_method must be 'circular_shift' or 'permute'.")

        low_list.append(np.percentile(null_r, alpha_low))
        high_list.append(np.percentile(null_r, alpha_high))

    centers = np.asarray(centers)
    r_real = np.asarray(r_real)
    p_vals = np.asarray(p_vals)
    low = np.asarray(low_list)
    high = np.asarray(high_list)

    in_null = (r_real >= low) & (r_real <= high)
    nonsig = p_vals > p_threshold
    r_condition = np.abs(r_real) < r_threshold

    condition = in_null & nonsig & r_condition

    counts = np.convolve(condition.astype(int), np.ones(K, dtype=int), mode="valid")
    hits = np.where(counts == K)[0]

    tau = float(centers[hits[0]]) if hits.size > 0 else np.nan

    return tau, centers, r_real, p_vals, low, high, condition


def tau_dec_from_accuracy(
    accuracy_original,
    accuracy_perturbed,
    window=20,
    step=1,
    n_null=500,
    ci=95,
    K=1,
    p_threshold=0.05,
    r_threshold=0.5,
    use_detrend=True,
    null_method="circular_shift",
):
    """
    Compute one accuracy-based decorrelation time per initial condition.

    For each initial condition, the decorrelation time is computed independently
    for each perturbed replica. The final value for that initial condition is the
    mean over perturbations, ignoring NaN values.

    Parameters
    ----------
    accuracy_original : numpy.ndarray
        Reference accuracy trajectories with shape ``(n_init, num_epochs)``.
    accuracy_perturbed : numpy.ndarray
        Perturbed accuracy trajectories with shape
        ``(n_init, num_pert, num_epochs)``.
    window, step, n_null, ci, K, p_threshold, r_threshold, use_detrend, null_method
        Parameters passed to ``compute_tau_corr_null_with_p``.

    Returns
    -------
    tau_decorr_CI_acc : numpy.ndarray
        One decorrelation time per initial condition, with shape ``(n_init,)``.
    """
    accuracy_original = np.asarray(accuracy_original)
    accuracy_perturbed = np.asarray(accuracy_perturbed)

    n_init = accuracy_perturbed.shape[0]
    num_pert = accuracy_perturbed.shape[1]

    tau_decorr_CI_acc = np.full(n_init, np.nan, dtype=float)

    for i in range(n_init):
        tau_aux = np.zeros(num_pert)

        for j in range(num_pert):
            tau, *_ = compute_tau_corr_null_with_p(
                accuracy_original[i],
                accuracy_perturbed[i, j],
                window=window,
                step=step,
                n_null=n_null,
                ci=ci,
                K=K,
                p_threshold=p_threshold,
                r_threshold=r_threshold,
                use_detrend=use_detrend,
                null_method=null_method,
            )

            tau_aux[j] = tau

        tau_decorr_CI_acc[i] = np.nanmean(tau_aux)

    return tau_decorr_CI_acc



def compute_z_infinity(embedding_reshaped, n_last=50, method="mean"):
    """
    Estimate the asymptotic embedded state ``z_infinity`` for each trajectory.

    Parameters
    ----------
    embedding_reshaped : numpy.ndarray
        Reshaped scalar embedding with shape ``(n_init, num_traj, L)``.
    n_last : int
        Number of final epochs used to estimate the asymptotic value.
    method : {"mean", "median", "last"}
        Estimator for the asymptotic state.

    Returns
    -------
    numpy.ndarray
        Estimated asymptotic states with shape ``(n_init, num_traj)``.
    """
    z = np.asarray(embedding_reshaped, dtype=float)

    if n_last <= 0:
        raise ValueError("n_last must be positive.")

    tail = z[:, :, -n_last:]

    if method == "mean":
        return np.nanmean(tail, axis=2)
    if method == "median":
        return np.nanmedian(tail, axis=2)
    if method == "last":
        return z[:, :, -1]

    raise ValueError("method must be 'mean', 'median', or 'last'.")


def compute_delta_from_z_infinity(z_infinity, descending=True):
    """
    Compute nearest-neighbor spacings between asymptotic embedded states.

    For each initial condition, the asymptotic states are ordered and the
    spacings are computed as

        Delta_i = z_(i-1)^infinity - z_(i)^infinity

    when ``descending=True``.

    Parameters
    ----------
    z_infinity : numpy.ndarray
        Asymptotic states with shape ``(n_init, num_traj)``.
    descending : bool
        If True, sort states from largest to smallest, matching the convention
        used in the manuscript notes.

    Returns
    -------
    deltas : numpy.ndarray
        Spacings with shape ``(n_init, num_traj - 1)``.
    z_sorted : numpy.ndarray
        Ordered asymptotic states with shape ``(n_init, num_traj)``.
    """
    z_infinity = np.asarray(z_infinity, dtype=float)

    if descending:
        z_sorted = np.sort(z_infinity, axis=1)[:, ::-1]
        deltas = z_sorted[:, :-1] - z_sorted[:, 1:]
    else:
        z_sorted = np.sort(z_infinity, axis=1)
        deltas = z_sorted[:, 1:] - z_sorted[:, :-1]

    return deltas, z_sorted


def preprocess_embedding_results(
    path,
    num_traj=None,
    num_epochs=None,
    n_last=50,
    z_inf_method="mean",
    compute_tau=True,
):
    """
    Complete post-processing pipeline for one saved ``.npz`` result file.

    This function is meant to be called after the expensive simulations are
    finished. It loads the data, reshapes the scalar embedding, computes
    embedded distances, estimates ``z_infinity``, obtains the spacings
    ``Delta``, and optionally computes both decorrelation times.

    Parameters
    ----------
    path : str
        Path to a saved ``.npz`` file.
    num_traj : int or None
        Number of trajectories per initial condition. If ``None``, it is
        inferred from ``accuracy_perturbed`` when available.
    num_epochs : int or None
        Number of training epochs. If ``None``, it is inferred from the saved
        embedding length.
    n_last : int
        Number of final epochs used to estimate ``z_infinity``.
    z_inf_method : {"mean", "median", "last"}
        Method used to estimate ``z_infinity``.
    compute_tau : bool
        If True, compute ``tau_dec`` from the embedding and from the accuracy.

    Returns
    -------
    dict
        Dictionary containing the loaded data and all derived quantities.
    """
    data = load_embedding_results(path)

    if num_traj is None:
        if "accuracy_perturbed" not in data:
            raise ValueError(
                "num_traj could not be inferred because accuracy_perturbed is missing."
            )
        num_traj = infer_num_trajectories_from_accuracy(data["accuracy_perturbed"])

    z = reshape_embedding(
        data["classic_mds"],
        num_traj=num_traj,
        num_epochs=num_epochs,
    )

    d_emb = embedding_distance_to_reference(z)
    z_infinity = compute_z_infinity(z, n_last=n_last, method=z_inf_method)
    delta, z_sorted = compute_delta_from_z_infinity(z_infinity, descending=True)

    processed = {
        "raw": data,
        "embedding": z,
        "embedding_distance": d_emb,
        "z_infinity": z_infinity,
        "z_infinity_sorted": z_sorted,
        "delta": delta,
    }

    if compute_tau:
        processed["tau_dec_embedding"] = tau_dec_from_embedding(z)

        if "accuracy_original" in data and "accuracy_perturbed" in data:
            processed["tau_dec_accuracy"] = tau_dec_from_accuracy(
                data["accuracy_original"],
                data["accuracy_perturbed"],
            )

    return processed

# -----------------------------------------------------------------------------
# Entry points
# -----------------------------------------------------------------------------

def main():
    """
    Standard SLURM-array entry point.

    ``SLURM_ARRAY_TASK_ID`` selects the learning rate from the list below.
    """
    learning_rate_array = [
        0.0005, 0.001, 0.005, 0.01,
        0.05, 0.1, 0.5, 1.0,
        2.5, 5.0, 7.5, 10.0,
        12.5, 15.0, 17.5, 20.0,
    ]

    slurm_array_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    learning_rate = learning_rate_array[slurm_array_task_id]

    lyapunov_embedding(learning_rate)


def main_1():
    """
    Fine learning-rate sweep from a common initial condition.

    This was used for the eta sweep where all simulations start from the same
    initial condition.
    """
    torch.manual_seed(0)

    net = MNISTNet()
    torch.save(net.state_dict(), "initial_state.pt")
    print("Saved initial state")

    learning_rate_array = np.round(np.arange(2.5, 20.1, 0.1), 1)

    slurm_array_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    learning_rate = learning_rate_array[slurm_array_task_id]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    initial_state = torch.load(os.path.join(base_dir, "initial_state.pt"), map_location="cpu")

    # Set to None to recover independent random initial conditions instead.
    #initial_state = None

    lyapunov_embedding_1(learning_rate, initial_state)


def main_2():
    """
    Sharded SLURM-array entry point.

    The SLURM task id encodes both the learning-rate index and the initial
    condition index. Each job writes one independent shard.
    """
    learning_rate_array = [
        0.0005, 0.001, 0.005, 0.01,
        0.05, 0.1, 0.5, 1.0,
        2.5, 5.0, 7.5, 10.0,
        12.5, 15.0, 17.5, 20.0,
    ]

    n_init = 10

    tid = int(os.environ["SLURM_ARRAY_TASK_ID"])
    lr_idx = tid // n_init
    init_idx = tid % n_init

    learning_rate = learning_rate_array[lr_idx]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    initial_state = torch.load(os.path.join(base_dir, "initial_state.pt"), map_location="cpu")
    initial_state_local = copy.deepcopy(initial_state)

    p_delta_embedding(learning_rate, init_idx, initial_state_local)


if __name__ == "__main__":
    # Choose the desired entry point:
    #   main()    -> standard learning-rate array
    #   main_1()  -> fine eta sweep from a common initial condition
    #   main_2()  -> sharded array over learning rates and initial conditions
    main()
    # main_1()
    # main_2()
