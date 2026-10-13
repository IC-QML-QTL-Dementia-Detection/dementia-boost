# Embeddings (Feature Maps) of the Qiskit Head

The quantum circuit of the Dressed Quantum Network has two parts: an **embedding** that loads the scaled features $\tilde{z}$ into a state $|\psi_e(\tilde{z})\rangle$, and a **variational block** that is the same for every embedding. The Qiskit head can use any of the embeddings below. The choice is the `ansatz` field of the run specification, so each embedding is its own configuration with its own `config_id`; `paper` is the default and the name inside every existing configuration ID. The PennyLane head implements only `paper`.

> [!NOTE]
> The embeddings other than `paper` have been run only with the Qiskit head (loss-level SPSA gradients). The PennyLane head, with exact adjoint gradients, has only been run with `paper`. A comparison between embeddings is therefore a comparison on one framework, and a comparison between frameworks exists only for `paper`.

This page is separate from `architecture.md` for now; the formulation of the dressed network around the circuit (pre-net, angle scaling, post-net, measurement) is in section 3.1 there.

## Implementation

Each embedding is a pair of functions in `qiskit_embeddings.py`: an angle function in PyTorch that maps $\tilde{z}$ to the gate angles, and the circuit part that applies those gates, one circuit parameter per angle. Computing the angles outside the circuit keeps every circuit parameter a plain input column (no products of run-time parameters inside the simulator), and autograd differentiates the angle function exactly back to the pre-net. `build_qiskit_ansatz` places the chosen embedding in front of the variational block, and `QiskitQuantumClassifierHead` applies the angle function after the $\tanh$ scaling. `QiskitQuantumLayer` only sees the angle columns.

```bash
# Train a Qiskit QTL head with a given embedding (default: paper); --seeds N runs the first N seeds
uv run scripts/training/train_qiskit_qtl_multiseed.py --ansatz zz_feature_map

# Evaluate only the configuration just trained (look its ID up with list_runs.py)
uv run scripts/metrics/evaluate_qiskit_qtl.py --config <config_id>
```

## Embeddings

With $n$ qubits and the scaled inputs $\tilde{z}_i \in [-\pi/2, \pi/2]$:

- **`paper` (angle embedding).** Every qubit goes to $|+\rangle$ and is then rotated about $z$ by its feature:
  $$|\psi_{\text{paper}}(\tilde{z})\rangle = \bigotimes_{i=1}^{n} R_z(\tilde{z}_i)\, H\, |0\rangle$$
  Each qubit sits on the equator of the Bloch sphere at azimuth $\tilde{z}_i$. The state is a product state, so the embedding creates no correlation between features. $n$ angles.

- **`rx`.** Every qubit is rotated about $x$ from $|0\rangle$, with no Hadamard:
  $$|\psi_{\text{rx}}(\tilde{z})\rangle = \bigotimes_{i=1}^{n} R_x(\tilde{z}_i)\, |0\rangle = \bigotimes_{i=1}^{n} \left(\cos\tfrac{\tilde{z}_i}{2}|0\rangle - i \sin\tfrac{\tilde{z}_i}{2}|1\rangle\right)$$
  The feature sets the polar angle of each qubit, so $\langle Z_i\rangle = \cos\tilde{z}_i$ before the variational block. Also a product state. $n$ angles.

- **`zz_feature_map`.** The ZZ feature map of Havlicek et al. as implemented by Qiskit's `ZZFeatureMap` (full entanglement, 2 repetitions). One repetition applies Hadamards, a phase on every qubit, and a $ZZ$ phase on every pair, with the pair angle a product of two features:
  $$U_{\Phi}(\tilde{z}) = \exp\Big(-i \sum_{(i,j)\in E} (\pi - \tilde{z}_i)(\pi - \tilde{z}_j)\, Z_i Z_j\Big)\, \exp\Big(-i \sum_{i} \tilde{z}_i Z_i\Big), \qquad |\psi_{\text{zz}}(\tilde{z})\rangle = \big(U_{\Phi}(\tilde{z})\, H^{\otimes n}\big)^{2} |0\rangle^{\otimes n}$$
  with $E$ every pair $i < j$ (15 pairs at $n = 6$). In gates, $\exp(-i \tilde{z}_i Z_i) = R_z(2\tilde{z}_i)$ and $\exp(-i \phi\, Z_i Z_j) = R_{zz}(2\phi)$, so the angles are $2\tilde{z}_i$ and $2(\pi - \tilde{z}_i)(\pi - \tilde{z}_j)$: $n + n(n-1)/2 = 21$ angles. Qiskit writes the same gates as $P$ and CNOT, $P$, CNOT; they differ from $R_z$ and $R_{zz}$ only by a global phase, so every $\langle Z_i\rangle$ is identical (checked numerically against `ZZFeatureMap`: state fidelity 1). The state is entangled, and each pair of features enters through a product term. The pair angles reach about $2(3\pi/2)^2 \approx 45$ rad, so the map is a very sensitive function of the inputs.

- **`zz_feature_map_linear`.** The same map with $E = \{(i, i+1)\}$, nearest neighbours only: 5 pairs and $n + (n-1) = 11$ angles at $n = 6$.

## Variational block (the same for every embedding)

For layer $l = 1, \dots, L$ with weights $\theta_{l,i}, \gamma_{l,i}, \beta_{l,i}$ (the three rows of `weights[l]` in PennyLane, the three vectors of the flat weight in Qiskit):

$$V_l = \Big[\prod_{i=0}^{n-1} \mathrm{CRY}_{(i+1) \bmod n \to i}(\beta_{l,i})\Big] \Big[\bigotimes_{i} R_z(\gamma_{l,i})\Big] \Big[\prod_{i=0}^{n-1} \mathrm{CNOT}_{i \to (i+1) \bmod n}\Big] \Big[\bigotimes_{i} R_z(\theta_{l,i})\Big]$$

Each product over $i$ is applied in increasing $i$ (the CNOT and CRY rings do not commute), and $\mathrm{CRY}_{c \to t}(\beta)$ rotates qubit $t$ about $y$ by $\beta$ when qubit $c$ is $|1\rangle$. The circuit state and the measurement are

$$|\phi_{\theta}(\tilde{z})\rangle = V_L \cdots V_1\, |\psi_e(\tilde{z})\rangle, \qquad \langle Z_i\rangle = \langle\phi_{\theta}(\tilde{z})|\, Z_i\, |\phi_{\theta}(\tilde{z})\rangle,$$

followed by the post-net. The weights number $3Ln$ (72 at $n = 6$, $L = 4$) and are initialised uniformly in $[-\pi, \pi]$. Every embedding is followed by the same $V_l$, so a difference between configurations comes from the embedding.

## Comparing embeddings

After training and evaluating each configuration, `scripts/viz/visualize_comparison.py` draws the metric distributions of all configurations side by side (test and validation), `scripts/viz/visualize_loss.py` overlays their validation loss curves, and `scripts/metrics/generate_improvement_report.py --config qiskit_qtl=<config_id> --output <file>` writes a comparative report for one chosen configuration without overwriting the standard one.
