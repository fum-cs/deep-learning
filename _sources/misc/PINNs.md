# Physics-Informed Neural Networks (PINNs)
## Final Session of Deep Learning Course

### Prerequisite Knowledge Check

Your students have already completed the harmonic oscillator problem using:
- **MLP (Multilayer Perceptron):** They experienced how standard neural networks fail to extrapolate beyond training data and overfit to noise.
- **GAN (Generative Adversarial Network):** They saw that while GANs can generate realistic samples, they don't inherently respect physical laws and require大量 training data.

Now we introduce **PINNs** – a paradigm that bridges the gap between data-driven learning and first-principles physics.

---

## Part 1: Motivation – Why PINNs?

### 1.1 The Fundamental Problem with Pure Data-Driven Approaches

Consider what your students discovered when solving the harmonic oscillator with MLP:

**Problem 1: Extrapolation Failure**
- An MLP trained on `t ∈ [0, 0.3]` cannot predict `t > 0.3` accurately
- The network learns a mapping, not the underlying law
- It has no "knowledge" that the oscillator should oscillate with a specific frequency and decay rate

**Problem 2: Noise Sensitivity**
- Real experimental data contains measurement noise
- MLPs overfit to this noise, learning spurious oscillations
- The resulting model violates energy conservation and other physical principles

**Problem 3: Data Scarcity**
- In many engineering contexts (aerospace, biomedical, climate science), high-quality data is expensive or impossible to obtain
- GANs require大量 data to generate realistic samples – a luxury we often don't have

### 1.2 What Makes PINNs Different?

The key insight of **Physics-Informed Neural Networks (PINNs)** is to **embed the governing differential equations directly into the loss function**.

Instead of learning only from data points:
```
Loss_data = MSE(predicted, measured)
```

PINNs learn from BOTH data AND physics:
```
Loss_PINN = Loss_data + λ × Loss_physics
```

Where `Loss_physics` measures how much the network's predictions violate the laws of physics.

### 1.3 The Analogy: Teaching a Student

Think of training a neural network like teaching a physics student:

| Approach | Analogy |
|----------|---------|
| **MLP** | Student only sees experimental results, never learns Newton's laws |
| **GAN** | Student learns to mimic experiments but doesn't understand why |
| **PINN** | Student learns both: "Here's what experiments show, AND here are the fundamental equations that must always be satisfied" |

---

## Part 2: Example 1 – Projectile Motion (Parabolic Trajectory)

### 2.1 Physics Background

**The Physical Scenario**

Imagine throwing a ball straight upward from a height of 1 meter with an initial velocity of 10 m/s. Gravity pulls it back down. Assuming no air resistance, what is its height at any time t?

**Governing Equation**

Newton's second law gives us:
$$F = ma = m \cdot \frac{d^2h}{dt^2} = -mg$$

Since mass cancels out:
$$\frac{d^2h}{dt^2} = -g$$

This is a **second-order ordinary differential equation (ODE)**. To solve it uniquely, we need two initial conditions:
1. Initial height: $h(0) = 1$ meter
2. Initial velocity: $\dot{h}(0) = 10$ m/s

**Analytical Solution**

Integrating twice gives the familiar parabolic equation:
$$h(t) = 1 + 10t - \frac{1}{2}(9.8)t^2$$

This is the **ground truth** we'll use to generate training data and evaluate our models.

### 2.2 Why This Example?

This problem is pedagogically valuable because:
1. **Simple physics** – Students already understand projectile motion from introductory physics
2. **Low-dimensional** – Only one input variable (time t) and one output (height h)
3. **Clear failure modes** – MLPs fail dramatically outside the training range
4. **Easy visualization** – We can plot everything in 2D

### 2.3 The Experimental Setup (Simulated)

Suppose you conduct an experiment and measure the ball's height at 5 different times during the first 0.5 seconds:

| Time (s) | Height (m) |
|----------|------------|
| 0.00 | 1.00 |
| 0.125 | 2.10 |
| 0.250 | 3.03 |
| 0.375 | 3.81 |
| 0.500 | 4.42 |

**Crucial observation:** Your measurements only go up to t = 0.5 s, but you want to predict the height at t = 1.5 s (which is 6.1 m in reality).

### 2.4 Standard MLP Approach (What Students Already Did)

A standard MLP would:
1. Take time t as input
2. Output predicted height ĥ(t)
3. Minimize mean squared error on the 5 data points

**The Result:** The MLP learns to fit the 5 points perfectly within t ∈ [0, 0.5], but outside this interval, its predictions are meaningless – it might go to zero, blow up, or oscillate randomly.

**Why does this happen?** The MLP has no constraints outside the training region. It doesn't "know" that gravity is still acting at t = 1.0 s.

### 2.5 PINN Solution – Adding Physics Knowledge

**Step 1: Reformulate the physics as a first-order ODE**

Instead of the second-order equation, we can write:
$$\frac{dh}{dt} = v(t) = 10 - 9.8t$$

This says: the velocity at any time equals the initial velocity minus gravity × time.

**Step 2: Define the physics loss**

At any set of time points `{t₁, t₂, ..., tₙ}`, we can:

- Compute the network's predicted height `ĥ(tᵢ)`
- Use automatic differentiation to compute `dĥ/dt` at those points
- Compare `dĥ/dt` to the known physical value `10 - 9.8tᵢ`
- The physics loss is the mean squared difference

**Step 3: Define the data loss**

At the 5 experimental time points, compute the MSE between predicted and measured heights.

**Step 4: Combine them**

$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \cdot \mathcal{L}_{physics}$$

The hyperparameter λ (e.g., λ = 0.01) balances the two objectives.

### 2.6 Automatic Differentiation – The Secret Sauce

How do we compute `dĥ/dt` from a neural network? PyTorch's **autograd** system does this automatically:

```python
# t_physics requires gradient tracking
t_physics = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], requires_grad=True)

# Forward pass: compute heights
h_pred = model(t_physics)

# Backward pass: compute dh/dt
dh_dt = torch.autograd.grad(h_pred, t_physics, 
                            grad_outputs=torch.ones_like(h_pred),
                            create_graph=True)[0]
```

This computes exact derivatives (not finite differences), making PINNs both accurate and computationally efficient.

### 2.7 Expected Results

After training, the PINN will:
- Fit the 5 data points accurately (low data loss)
- Also satisfy `dh/dt = 10 - 9.8t` everywhere (low physics loss)
- Successfully predict heights at t = 1.5 s and beyond

**The key insight:** The physics loss acts as a **regularizer** that constrains the network's behavior even in regions without data.

### 2.8 Complete Code with Detailed Comments

```python
"""
PINN for Projectile Motion
Problem: Predict height h(t) of a ball thrown upward
Physics: dh/dt = v0 - g*t
Data: 5 measurements in [0, 0.5] seconds
Goal: Extrapolate to t = 2.0 seconds
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# PHYSICAL PARAMETERS
# ============================================
g = 9.8          # gravitational acceleration (m/s²)
h0 = 1.0         # initial height (m)
v0 = 10.0        # initial velocity (m/s)

def true_solution(t):
    """Analytical solution for reference"""
    return h0 + v0 * t - 0.5 * g * t**2

# ============================================
# GENERATE TRAINING DATA (simulated experiment)
# ============================================
# We only have measurements in the first 0.5 seconds
t_experiment = torch.linspace(0, 0.5, 5).view(-1, 1)
h_experiment = true_solution(t_experiment)

# For visualization, we'll use the full range [0, 2]
t_full = torch.linspace(0, 2, 200).view(-1, 1)
h_full = true_solution(t_full)

# ============================================
# NEURAL NETWORK ARCHITECTURE
# ============================================
class ProjectilePINN(nn.Module):
    """
    Simple feedforward network with 2 hidden layers
    Input: time t (scalar)
    Output: height h(t) (scalar)
    """
    def __init__(self, hidden_units=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_units),
            nn.Tanh(),                    # Smooth activation for derivatives
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 1)
        )
    
    def forward(self, t):
        return self.net(t)

def compute_derivative(y, t):
    """
    Compute dy/dt using PyTorch's automatic differentiation
    y: output tensor
    t: input tensor (requires grad)
    """
    return torch.autograd.grad(
        y, t,
        grad_outputs=torch.ones_like(y),
        create_graph=True      # Needed for higher-order derivatives
    )[0]

# ============================================
# TRAINING FUNCTION
# ============================================
def train_pinn(model, t_data, h_data, t_physics, 
               lambda_physics=0.01, epochs=3000, lr=0.001):
    """
    Train a PINN with combined data and physics loss
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Storage for tracking loss history
    loss_history = {'total': [], 'data': [], 'physics': []}
    
    # Check if physics points are provided
    use_physics = t_physics.numel() > 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # ----- DATA LOSS -----
        # Fit the experimental measurements
        h_pred_data = model(t_data)
        loss_data = torch.mean((h_pred_data - h_data)**2)
        
        # ----- PHYSICS LOSS -----
        if use_physics:
            # Ensure dh/dt follows Newton's law
            t_physics.requires_grad_(True)
            h_physics = model(t_physics)
            dh_dt = compute_derivative(h_physics, t_physics)
            
            # Known physics: dh/dt = v0 - g*t
            dh_dt_true = v0 - g * t_physics
            loss_physics = torch.mean((dh_dt - dh_dt_true)**2)
        else:
            loss_physics = torch.tensor(0.0)
        
        # ----- TOTAL LOSS -----
        loss_total = loss_data + lambda_physics * loss_physics
        
        # Backpropagation
        loss_total.backward()
        optimizer.step()
        
        # Record losses
        loss_history['total'].append(loss_total.item())
        loss_history['data'].append(loss_data.item())
        loss_history['physics'].append(loss_physics.item())
        
        # Print progress
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1:4d}: "
                  f"Total={loss_total.item():.6f}, "
                  f"Data={loss_data.item():.6f}, "
                  f"Physics={loss_physics.item():.6f}")
    
    return loss_history

# ============================================
# TRAIN BOTH MODELS FOR COMPARISON
# ============================================
# Standard MLP (data only)
print("=" * 50)
print("TRAINING STANDARD MLP (NO PHYSICS)")
print("=" * 50)
mlp = ProjectilePINN()
loss_history_mlp = train_pinn(
    mlp, t_experiment, h_experiment, 
    t_physics=torch.tensor([]),  # No physics points
    lambda_physics=0.0,          # Physics loss disabled
    epochs=3000
)

# PINN (data + physics)
print("\n" + "=" * 50)
print("TRAINING PINN (WITH PHYSICS)")
print("=" * 50)
# Use 50 points spread across the whole domain for physics loss
t_physics = torch.linspace(0, 2, 50).view(-1, 1)
pinn = ProjectilePINN()
loss_history_pinn = train_pinn(
    pinn, t_experiment, h_experiment,
    t_physics, lambda_physics=0.01,
    epochs=3000
)

# ============================================
# VISUALIZE RESULTS
# ============================================
with torch.no_grad():
    h_mlp = mlp(t_full)
    h_pinn = pinn(t_full)

plt.figure(figsize=(14, 5))

# Plot 1: Predictions
plt.subplot(1, 2, 1)
plt.plot(t_full, h_full, 'k--', linewidth=2, 
         label='True Solution (Physics)')
plt.plot(t_full, h_mlp, 'b-', linewidth=2, alpha=0.7,
         label='MLP (Data only)')
plt.plot(t_full, h_pinn, 'g-', linewidth=2, alpha=0.7,
         label='PINN (Data + Physics)')
plt.scatter(t_experiment, h_experiment, color='red', s=80,
           label='Experimental Data (5 points)', zorder=5)
plt.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5,
           label='Training region boundary')
plt.xlabel('Time t (seconds)', fontsize=12)
plt.ylabel('Height h(t) (meters)', fontsize=12)
plt.title('PINN vs MLP: Extrapolation to t=2.0s', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.xlim(-0.05, 2.05)
plt.ylim(-1, 12)

# Plot 2: Loss history
plt.subplot(1, 2, 2)
plt.semilogy(loss_history_pinn['total'], 'g-', alpha=0.7, label='PINN Total')
plt.semilogy(loss_history_pinn['data'], 'g--', alpha=0.5, label='PINN Data')
plt.semilogy(loss_history_pinn['physics'], 'g:', alpha=0.5, label='PINN Physics')
plt.semilogy(loss_history_mlp['total'], 'b-', alpha=0.7, label='MLP Total')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (log scale)', fontsize=12)
plt.title('Training Convergence', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# QUANTITATIVE COMPARISON
# ============================================
print("\n" + "=" * 50)
print("QUANTITATIVE RESULTS")
print("=" * 50)

# Test at t=1.5s (far outside training)
t_test = torch.tensor([[1.5]])
h_true = true_solution(t_test).item()

with torch.no_grad():
    h_mlp_test = mlp(t_test).item()
    h_pinn_test = pinn(t_test).item()

print(f"Test point: t = 1.5 s")
print(f"True height:        {h_true:.4f} m")
print(f"MLP prediction:     {h_mlp_test:.4f} m (error: {abs(h_true-h_mlp_test):.4f})")
print(f"PINN prediction:    {h_pinn_test:.4f} m (error: {abs(h_true-h_pinn_test):.4f})")
```

### 2.9 Discussion Questions for Students

1. **Why does the MLP fail at extrapolation?**  
   *Answer: It has no constraints outside training region; minimizes only data loss.*

2. **How does automatic differentiation enable PINNs?**  
   *Answer: It computes exact derivatives of network output with respect to inputs, allowing us to enforce differential equations.*

3. **What happens if λ (physics weight) is too large?**  
   *Answer: The network will perfectly satisfy physics but ignore data completely – it becomes a pure ODE solver.*

4. **What if λ is too small?**  
   *Answer: The network behaves like a standard MLP, overfitting to noise and failing to extrapolate.*

---

## Part 3: Example 2 – Projectile Motion with Noisy Data

### 3.1 The Problem with Real Experiments

In the previous example, we used perfect (noise-free) data. But real experiments always have **measurement uncertainty**:

- Sensor precision limits (e.g., ±2 cm for a ultrasonic sensor)
- Environmental disturbances (wind, vibration, temperature drift)
- Human error in recording

Let's add realistic noise to our measurements and observe how MLPs and PINNs respond.

### 3.2 Noise Characteristics

We'll add **Gaussian (normal) noise** with:
- Mean = 0 (no systematic bias)
- Standard deviation = 0.3 meters (about 30 cm of uncertainty)

For a ball reaching 6 meters height, this represents ~5% measurement error – quite realistic.

### 3.3 What Happens to Standard MLP?

**The Overfitting Problem:**

An MLP trained on noisy data will:
- Try to pass exactly through every noisy point
- Learn the noise pattern as if it were real signal
- Create a "wiggly" trajectory that violates physics
- Fail dramatically when asked to predict new points

**Why this matters:** In scientific computing, we want models that capture the **underlying physical law**, not the measurement errors.

### 3.4 How PINNs Handle Noise

The physics loss acts as a **powerful regularizer**:

- The network cannot simultaneously fit the noisy data AND satisfy `dh/dt = v0 - g*t`
- It must find a compromise: follow physics while staying reasonably close to measurements
- The result is a smooth trajectory that **ignores the noise** and follows the true physics

**Mathematical intuition:** The physics loss imposes that the solution must lie in the subspace of functions satisfying the ODE. Noisy data points outside this subspace are "projected" onto it.

### 3.5 Extended Code with Noise Analysis

```python
"""
PINN for Projectile Motion with Noisy Measurements
Demonstrates robustness to experimental uncertainty
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# PHYSICAL PARAMETERS (same as before)
# ============================================
g, h0, v0 = 9.8, 1.0, 10.0

def true_solution(t):
    return h0 + v0 * t - 0.5 * g * t**2

# ============================================
# GENERATE NOISY EXPERIMENTAL DATA
# ============================================
torch.manual_seed(42)  # Reproducible noise

t_experiment = torch.linspace(0, 0.8, 12).view(-1, 1)  # More points, still limited
h_clean = true_solution(t_experiment)

# Add Gaussian noise (std = 0.3 m)
noise_std = 0.3
noise = noise_std * torch.randn_like(h_clean)
h_noisy = h_clean + noise

print(f"Added noise with standard deviation: {noise_std:.2f} m")
print(f"Noise range: [{noise.min().item():.3f}, {noise.max().item():.3f}] m")

# ============================================
# NETWORK ARCHITECTURE (same)
# ============================================
class NoisyProjectilePINN(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, t):
        return self.net(t)

def derivative(y, t):
    return torch.autograd.grad(y, t, torch.ones_like(y), 
                               create_graph=True)[0]

# ============================================
# TRAINING WITH DIFFERENT λ VALUES
# ============================================
t_full = torch.linspace(0, 2, 200).view(-1, 1)
h_full = true_solution(t_full)
t_physics = torch.linspace(0, 2, 60).view(-1, 1)

# Try different physics weights
lambda_values = [0.0, 0.001, 0.01, 0.1]
models = {}
losses = {}

for lam in lambda_values:
    print(f"\n{'='*50}")
    print(f"Training with λ_physics = {lam}")
    print(f"{'='*50}")
    
    model = NoisyProjectilePINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    loss_history = {'total': [], 'data': [], 'physics': []}
    
    for epoch in range(4000):
        optimizer.zero_grad()
        
        # Data loss (on noisy measurements)
        h_pred_data = model(t_experiment)
        loss_data = torch.mean((h_pred_data - h_noisy)**2)
        
        # Physics loss (using clean physics)
        t_physics.requires_grad_(True)
        h_phys = model(t_physics)
        dh_dt = derivative(h_phys, t_physics)
        dh_dt_true = v0 - g * t_physics
        loss_physics = torch.mean((dh_dt - dh_dt_true)**2)
        
        # Total loss
        loss_total = loss_data + lam * loss_physics
        
        loss_total.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1:4d}: Total={loss_total.item():.6f}, "
                  f"Data={loss_data.item():.6f}, Physics={loss_physics.item():.6f}")
        
        # Store loss history
        if epoch % 100 == 0:
            loss_history['total'].append(loss_total.item())
            loss_history['data'].append(loss_data.item())
            loss_history['physics'].append(loss_physics.item())
    
    models[lam] = model
    losses[lam] = loss_history

# ============================================
# VISUALIZATION
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, lam in enumerate(lambda_values):
    ax = axes[idx]
    model = models[lam]
    
    with torch.no_grad():
        h_pred = model(t_full)
    
    # Plot predictions
    ax.plot(t_full, h_full, 'k--', linewidth=2, label='True Physics')
    ax.plot(t_full, h_pred, 'b-', linewidth=2, alpha=0.7, 
            label='PINN Prediction')
    ax.scatter(t_experiment, h_noisy, color='red', s=40, alpha=0.6,
              label='Noisy Data', zorder=5)
    ax.axvline(x=0.8, color='gray', linestyle=':', alpha=0.5)
    
    # Calculate MSE on clean data (evaluation metric)
    with torch.no_grad():
        mse_vs_true = torch.mean((h_pred - h_full)**2).item()
    
    ax.set_title(f'λ_physics = {lam}\nMSE vs True = {mse_vs_true:.4f}', 
                 fontsize=12)
    ax.set_xlabel('Time t (seconds)')
    ax.set_ylabel('Height h(t) (meters)')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(-1, 12)

plt.tight_layout()
plt.suptitle("Effect of Physics Loss Weight on Noise Robustness", 
             fontsize=14, y=1.02)
plt.show()

# ============================================
# LOSS CONVERGENCE COMPARISON
# ============================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
for lam in lambda_values:
    plt.semilogy(losses[lam]['total'], label=f'λ={lam}')
plt.xlabel('Epoch (×100)', fontsize=12)
plt.ylabel('Total Loss (log scale)', fontsize=12)
plt.title('Training Convergence for Different λ Values', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for lam in [0.001, 0.01, 0.1]:
    if lam in losses:
        data_losses = losses[lam]['data']
        plt.semilogy(data_losses, label=f'λ={lam} (data loss)')
# Also plot MLP baseline (λ=0)
plt.semilogy(losses[0.0]['data'], 'k-', linewidth=2, label='λ=0 (MLP baseline)')
plt.xlabel('Epoch (×100)', fontsize=12)
plt.ylabel('Data Loss (log scale)', fontsize=12)
plt.title('Data Loss Convergence (lower is better for fitting data)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# QUANTITATIVE ANALYSIS
# ============================================
print("\n" + "=" * 60)
print("QUANTITATIVE COMPARISON ON NOISY DATA")
print("=" * 60)
print(f"{'λ_physics':<12} {'MSE_vs_clean':<15} {'MSE_vs_noisy':<15} {'Physics_loss':<15}")
print("-" * 60)

for lam in lambda_values:
    model = models[lam]
    with torch.no_grad():
        h_pred_full = model(t_full)
        h_pred_train = model(t_experiment)
        
        mse_vs_clean = torch.mean((h_pred_full - h_full)**2).item()
        mse_vs_noisy = torch.mean((h_pred_train - h_noisy)**2).item()
    
    # Get final physics loss
    t_physics.requires_grad_(True)
    with torch.enable_grad():
        h_phys = model(t_physics)
        dh_dt = derivative(h_phys, t_physics)
        dh_dt_true = v0 - g * t_physics
        physics_loss = torch.mean((dh_dt - dh_dt_true)**2).item()
    
    print(f"{lam:<12.3f} {mse_vs_clean:<15.6f} {mse_vs_noisy:<15.6f} {physics_loss:<15.8f}")

print("\n" + "=" * 60)
print("KEY OBSERVATIONS:")
print("=" * 60)
print("1. λ = 0 (MLP): Lowest data loss on training data, but highest error vs true solution")
print("2. λ = 0.01-0.1: Higher data loss (good! ignoring noise), lower error vs true solution")
print("3. Very large λ: Physics dominates, data is almost ignored")
```

### 3.6 Key Pedagogical Observations for Students

| λ Value | Behavior | Best for... |
|---------|----------|--------------|
| 0 (MLP) | Fits noise exactly, poor extrapolation | When data is perfect (never happens) |
| 0.001 | Still overfits, slightly better | Low noise scenarios |
| 0.01 | **Sweet spot** – balances physics and data | Most real experiments |
| 0.1 | Physically accurate, ignores some data | High noise, well-understood physics |

**The key insight:** There's a trade-off controlled by λ. Students should learn to tune this hyperparameter using validation data (hold-out clean points if available) or physical reasoning.

---

## Part 4: Example 3 – Damped Harmonic Oscillator

### 4.1 Physics Background

**Why This Problem?**

Your students already attempted this with MLP and GAN. They likely encountered:
- MLP: Can't capture oscillations outside training window
- GAN: Generates plausible-looking but physically impossible trajectories (e.g., energy increasing)

Now they'll see how PINN solves both problems simultaneously.

**The Physical System**

Consider a mass attached to a spring, immersed in a viscous fluid (like water or oil):
- Spring pulls back proportionally to displacement (Hooke's law: F = -kx)
- Fluid resists motion proportionally to velocity (F = -μ dx/dt)
- Mass accelerates according to F = ma

**Governing Equation (Second-Order ODE)**

$$m\frac{d^2x}{dt^2} + \mu\frac{dx}{dt} + kx = 0$$

**Standard Form**

Dividing by m:
$$\frac{d^2x}{dt^2} + 2\delta\frac{dx}{dt} + \omega_0^2 x = 0$$

where:
- $\delta = \frac{\mu}{2m}$ = damping coefficient (controls energy loss rate)
- $\omega_0 = \sqrt{\frac{k}{m}}$ = natural frequency (how fast it would oscillate without damping)

**Initial Conditions**
- $x(0) = 1$ (pull mass to position 1 and release)
- $\dot{x}(0) = 0$ (release from rest)

**Underdamped Case ($\delta < \omega_0$) – What We'll Solve**

When damping is light, the system oscillates with exponentially decaying amplitude:
$$x(t) = e^{-\delta t}\left(\cos(\omega t) + \frac{\delta}{\omega}\sin(\omega t)\right)$$

where $\omega = \sqrt{\omega_0^2 - \delta^2}$ is the **damped natural frequency**.

**Parameters for This Example**
- $\delta = 2$ (moderate damping)
- $\omega_0 = 20$ (fast oscillation)
- $\omega = \sqrt{400 - 4} = \sqrt{396} \approx 19.9$

The solution will oscillate about 3-4 times before decaying significantly.

### 4.2 Why This Challenged MLP and GAN

**MLP Difficulties:**
1. **High frequency** ($\omega \approx 20$) requires many training points to resolve oscillations
2. **Exponential decay** is not naturally represented by standard activations (ReLU, sigmoid)
3. **Extrapolation** beyond training window is impossible – MLP doesn't "know" oscillations continue

**GAN Difficulties:**
1. **Mode collapse** – GAN might generate only a few oscillation patterns
2. **No physics constraints** – Generated trajectories may violate energy conservation
3. **Training instability** – Adversarial training is notoriously hard

### 4.3 PINN Solution Strategy

PINNs address these challenges through:

**1. Physics Loss with Second Derivatives**

We need to enforce the second-order ODE. Using automatic differentiation twice:
```python
x = model(t)
dx_dt = gradient(x, t)      # first derivative
d2x_dt2 = gradient(dx_dt, t) # second derivative
residual = d2x_dt2 + 2*δ*dx_dt + ω0²*x
loss_physics = mean(residual²)
```

**2. Initial Condition Loss**

We must enforce both initial conditions:
```python
x0 = model(t=0)
dx0_dt = gradient(x0, t=0)
loss_ic = (x0 - 1)² + (dx0_dt - 0)²
```

**3. Data Loss (if available)**

If we have some measurements (even noisy ones), we add:
```python
loss_data = mean((model(t_data) - x_data)²)
```

### 4.4 Complete Implementation (Based on Your File)

```python
"""
PINN for Damped Harmonic Oscillator
Based on the code from benmoseley/harmonic-oscillator-pinn
Extended with detailed explanations for teaching
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================
# PHYSICAL PARAMETERS
# ============================================
delta = 2.0      # damping coefficient (controls energy loss)
omega0 = 20.0    # natural frequency (controls oscillation speed)
omega = np.sqrt(omega0**2 - delta**2)  # damped frequency

# Parameters for the ODE: d²x/dt² + 2δ dx/dt + ω0² x = 0
mu_physics = 2 * delta     # coefficient for dx/dt
k_physics = omega0**2      # coefficient for x

print("=" * 60)
print("PHYSICAL PARAMETERS")
print("=" * 60)
print(f"δ (damping coefficient) = {delta}")
print(f"ω₀ (natural frequency) = {omega0}")
print(f"ω (damped frequency) = {omega:.4f}")
print(f"Expected oscillations in [0,1]: about {omega/(2*np.pi):.1f} cycles")

def exact_solution(t):
    """Analytical solution for underdamped case"""
    return torch.exp(-delta * t) * (torch.cos(omega * t) + 
                                    (delta/omega) * torch.sin(omega * t))

# ============================================
# GENERATE "EXPERIMENTAL" DATA
# ============================================
# Full domain for visualization
t_full = torch.linspace(0, 1, 500).view(-1, 1)
x_full = exact_solution(t_full)

# Training data: sparse measurements (only 10 points!)
t_data = t_full[0:200:20]  # First 10 points only (t ∈ [0, 0.4])
x_data = exact_solution(t_data)

print(f"\nTraining data: {len(t_data)} points in t ∈ [0, {t_data[-1].item():.2f}]")
print(f"We will test extrapolation to t = 1.0 (no data beyond {t_data[-1].item():.2f})")

# ============================================
# NETWORK ARCHITECTURE (from your file)
# ============================================
class HarmonicOscillatorPINN(nn.Module):
    """
    Fully Connected Network with Tanh activation
    Architecture: Input → Hidden → ... → Hidden → Output
    """
    def __init__(self, n_input=1, n_output=1, n_hidden=32, n_layers=3):
        super().__init__()
        activation = nn.Tanh  # Tanh is smooth and differentiable
        
        # Input layer
        self.fcs = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            activation()
        )
        
        # Hidden layers (n_layers - 1 of them)
        self.fch = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                activation()
            ) for _ in range(n_layers - 1)
        ])
        
        # Output layer
        self.fce = nn.Linear(n_hidden, n_output)
    
    def forward(self, t):
        t = self.fcs(t)
        t = self.fch(t)
        t = self.fce(t)
        return t

# ============================================
# STANDARD MLP TRAINING (Data Only)
# ============================================
print("\n" + "=" * 60)
print("TRAINING STANDARD MLP (DATA ONLY)")
print("=" * 60)

mlp = HarmonicOscillatorPINN()
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)

for epoch in range(3000):
    optimizer_mlp.zero_grad()
    x_pred = mlp(t_data)
    loss = torch.mean((x_pred - x_data)**2)
    loss.backward()
    optimizer_mlp.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1:4d}: Loss = {loss.item():.8f}")

# ============================================
# PINN TRAINING (Data + Physics)
# ============================================
print("\n" + "=" * 60)
print("TRAINING PINN (DATA + PHYSICS)")
print("=" * 60)

pinn = HarmonicOscillatorPINN()
optimizer_pinn = torch.optim.Adam(pinn.parameters(), lr=1e-4)

# Points where we enforce physics (50 points across full domain)
t_physics = torch.linspace(0, 1, 50).view(-1, 1).requires_grad_(True)

# Initial condition points
t0 = torch.zeros(1, 1, requires_grad=True)

# Weight for physics loss (tuned empirically)
lambda_physics = 1e-4
lambda_ic = 0.1  # Weight for initial condition loss

for epoch in range(15000):
    optimizer_pinn.zero_grad()
    
    # ----- DATA LOSS -----
    x_pred_data = pinn(t_data)
    loss_data = torch.mean((x_pred_data - x_data)**2)
    
    # ----- PHYSICS LOSS (second-order ODE) -----
    # Forward pass on physics points
    x_phys = pinn(t_physics)
    
    # First derivative (velocity)
    dx_dt = torch.autograd.grad(
        x_phys, t_physics,
        grad_outputs=torch.ones_like(x_phys),
        create_graph=True
    )[0]
    
    # Second derivative (acceleration)
    d2x_dt2 = torch.autograd.grad(
        dx_dt, t_physics,
        grad_outputs=torch.ones_like(dx_dt),
        create_graph=True
    )[0]
    
    # Residual of the ODE: should be zero
    # d²x/dt² + 2δ dx/dt + ω₀² x = 0
    residual = d2x_dt2 + mu_physics * dx_dt + k_physics * x_phys
    loss_physics = torch.mean(residual**2)
    
    # ----- INITIAL CONDITION LOSS -----
    x0_pred = pinn(t0)
    dx0_dt = torch.autograd.grad(
        x0_pred, t0,
        grad_outputs=torch.ones_like(x0_pred),
        create_graph=True
    )[0]
    
    loss_ic = (x0_pred - 1.0)**2 + (dx0_dt - 0.0)**2
    
    # ----- TOTAL LOSS -----
    loss = loss_data + lambda_physics * loss_physics + lambda_ic * loss_ic
    
    loss.backward()
    optimizer_pinn.step()
    
    if (epoch + 1) % 3000 == 0:
        print(f"Epoch {epoch+1:5d}: "
              f"Total={loss.item():.6f}, "
              f"Data={loss_data.item():.6f}, "
              f"Physics={loss_physics.item():.8f}, "
              f"IC={loss_ic.item():.8f}")

# ============================================
# VISUALIZATION
# ============================================
with torch.no_grad():
    x_mlp_pred = mlp(t_full)
    x_pinn_pred = pinn(t_full)

plt.figure(figsize=(14, 5))

# Plot 1: Time domain predictions
plt.subplot(1, 2, 1)
plt.plot(t_full, x_full, 'k--', linewidth=2, label='Exact Solution')
plt.plot(t_full, x_mlp_pred, 'b-', linewidth=2, alpha=0.7, 
         label='MLP (Data only)')
plt.plot(t_full, x_pinn_pred, 'g-', linewidth=2, alpha=0.7, 
         label='PINN (Data + Physics)')
plt.scatter(t_data, x_data, color='red', s=60, 
           label='Training Data (10 points)', zorder=5)
plt.axvline(x=t_data[-1].item(), color='gray', linestyle=':', alpha=0.7,
           label='Training region boundary')
plt.xlabel('Time t (seconds)', fontsize=12)
plt.ylabel('Displacement x(t)', fontsize=12)
plt.title('PINN vs MLP: Damped Harmonic Oscillator', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.xlim(-0.02, 1.02)
plt.ylim(-0.8, 1.1)

# Plot 2: Zoom on extrapolation region
plt.subplot(1, 2, 2)
mask = t_full.squeeze() > 0.4
t_extrap = t_full[mask]
x_exact_extrap = x_full[mask]
x_mlp_extrap = x_mlp_pred[mask]
x_pinn_extrap = x_pinn_pred[mask]

plt.plot(t_extrap, x_exact_extrap, 'k--', linewidth=2, label='Exact')
plt.plot(t_extrap, x_mlp_extrap, 'b-', linewidth=2, alpha=0.7, label='MLP')
plt.plot(t_extrap, x_pinn_extrap, 'g-', linewidth=2, alpha=0.7, label='PINN')
plt.axvline(x=t_data[-1].item(), color='gray', linestyle=':', alpha=0.7,
           label='Last training point')
plt.xlabel('Time t (seconds) - Extrapolation Region', fontsize=12)
plt.ylabel('Displacement x(t)', fontsize=12)
plt.title('Zoom: Extrapolation Performance (t > 0.4)', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.xlim(0.4, 1.02)
plt.ylim(-0.8, 0.6)

plt.tight_layout()
plt.show()

# ============================================
# QUANTITATIVE COMPARISON
# ============================================
print("\n" + "=" * 60)
print("QUANTITATIVE RESULTS")
print("=" * 60)

# Compute MSE on full domain
mse_mlp = torch.mean((x_mlp_pred - x_full)**2).item()
mse_pinn = torch.mean((x_pinn_pred - x_full)**2).item()

print(f"Mean Squared Error on t ∈ [0, 1]:")
print(f"  MLP:  {mse_mlp:.6f}")
print(f"  PINN: {mse_pinn:.6f}")

# Compute at specific test points
test_times = [0.6, 0.8, 1.0]
print(f"\nPredictions at specific test times:")
print(f"{'t':<6} {'Exact':<10} {'MLP':<12} {'PINN':<12} {'MLP Error':<12} {'PINN Error':<12}")
print("-" * 65)

with torch.no_grad():
    for t_val in test_times:
        t_tensor = torch.tensor([[t_val]])
        exact = exact_solution(t_tensor).item()
        mlp_pred = mlp(t_tensor).item()
        pinn_pred = pinn(t_tensor).item()
        
        print(f"{t_val:<6.1f} {exact:<10.4f} {mlp_pred:<12.4f} {pinn_pred:<12.4f} "
              f"{abs(exact-mlp_pred):<12.4f} {abs(exact-pinn_pred):<12.4f}")
```

### 4.5 Understanding the Results

**What Students Will Observe:**

1. **MLP Performance:**
   - Fits training data perfectly (low data loss)
   - Completely wrong for t > 0.5
   - Predictions might blow up, decay too fast, or oscillate at wrong frequency

2. **PINN Performance:**
   - Slightly higher error on training data (acceptable trade-off)
   - Accurately predicts oscillations beyond training region
   - Correctly captures both frequency (ω ≈ 20) and decay rate (δ = 2)

**Why PINN Succeeds Where MLP Failed:**

The physics loss constrains the network to a very specific **function space** – the set of functions that satisfy the ODE. Even with minimal data, the network "knows" the solution must oscillate with frequency ω and decay like e^{-δt}.

### 4.6 Comparison with GAN Results (Connection to Previous Assignment)

Students previously attempted this with GANs. Let's highlight the advantages of PINN over GAN:

| Aspect | GAN | PINN |
|--------|-----|------|
| Training stability | Unstable (mode collapse, non-convergence) | Stable (standard gradient descent) |
| Data requirement | Large dataset needed | Works with tiny datasets |
| Physics guarantee | None | Enforced via loss function |
| Interpretability | Black box | Can extract residual, derivatives |
| Determinism | Different seeds → different results | Consistent given same initialization |

---

## Part 5: Summary and Course Conclusion

### 5.1 The PINN Framework – General Recipe

For any physical system, implementing a PINN involves four steps:

```
1. Define the neural network (input: coordinates/times, output: solution)
2. Define the governing differential equations (physics)
3. Compute residuals using automatic differentiation
4. Minimize combined loss: Data + Physics + (Initial/Boundary conditions)
```

### 5.2 When to Use PINNs (vs. Traditional Methods)

| Scenario | Recommended approach |
|----------|---------------------|
| Abundant clean data | MLP or any standard method |
| No data, known physics | Traditional numerical solvers (FEM, FDM) |
| **Little data, known physics** | **PINN (sweet spot!)** |
| Noisy data, known physics | PINN with tuned λ |
| Partially known physics | PINN with incomplete model |
| Inverse problems (find parameters) | PINN (differentiate through physics) |

### 5.3 Limitations of PINNs (Be Honest with Students)

1. **Training can be slow** – Physics loss adds computational cost
2. **Hyperparameter tuning** – Finding the right λ requires experimentation
3. **Stiff ODEs/PDEs** – Fast dynamics are hard to learn (spectral bias)
4. **High-dimensional problems** – Curse of dimensionality still applies

### 5.4 The Big Picture: Where We've Been in This Course

```
Week 1-2:   Regression & MLP (function approximation basics)
Week 3-4:   CNNs (spatial structure, image data)
Week 5-6:   GANs (generative modeling, adversarial learning)
Week 7:     RNNs/LSTMs (sequential data)
Week 8:     Transformers & Attention
Week 9-10:  Autoencoders & Latent Representations
Week 11:    Reinforcement Learning
Week 12:    PINNs (integrating physics into deep learning) ← TODAY
```

**The Unifying Theme:** All these methods learn from data. PINNs add prior knowledge (physics) to the learning process, bridging the gap between data-driven and knowledge-driven AI.

### 5.5 Final Thoughts for Students

**Quote to End the Course:**

> *"In the end, deep learning is not about replacing physical laws with black boxes. It's about using neural networks as flexible function approximators that can respect both data AND first principles. PINNs represent a paradigm shift: from pure pattern recognition to physics-aware learning. As you move forward in your careers, remember that the most powerful AI systems will be those that combine the best of both worlds: the flexibility of data-driven learning and the reliability of physical laws."*

### 5.6 Take-Home Assignment (Optional Extension)

Ask students to modify the oscillator code to:

1. **Change the damping** – Set δ = 5 (overdamped case) and update the physics loss accordingly
2. **Add boundary conditions** – Instead of ICs, try x(0)=1 and x(1)=0 (two-point boundary value problem)
3. **Inverse problem** – Treat δ and ω₀ as unknown parameters to be learned from sparse data

---

## Appendix: Quick Reference Card for Students

```python
# PINN Template (General Structure)
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Define network layers with Tanh activation
        self.net = nn.Sequential(...)
    
    def forward(self, x):
        return self.net(x)

# Training loop pseudocode
model = PINN()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    # Data loss (if data available)
    loss_data = MSE(model(x_data), y_data)
    
    # Physics loss (using automatic differentiation)
    u = model(x_physics)
    u_x = torch.autograd.grad(u, x_physics, ...)[0]
    u_xx = torch.autograd.grad(u_x, x_physics, ...)[0]
    residual = u_xx + ...  # Your PDE/ODE here
    loss_physics = MSE(residual, 0)
    
    # Initial/Boundary conditions
    loss_ic = MSE(model(x_boundary), u_boundary)
    
    # Total loss
    loss = loss_data + λ_p * loss_physics + λ_ic * loss_ic
    loss.backward()
    optimizer.step()
```

