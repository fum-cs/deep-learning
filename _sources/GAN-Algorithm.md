1. INITIALIZATION
   - Set device (CPU/GPU)
   - Define latent dimension z_dim = 5
   - Define batch size = 32
   - Initialize Generator G (z_dim → 2D output)
   - Initialize Discriminator D (2D input → 1D output [0,1])
   - Initialize optimizers (Adam, lr=0.0002, betas=(0.5, 0.999))
   - Define loss function (Binary Cross Entropy)

2. FOR each training epoch (1 to 15000):
   
   // Step 1: Train Discriminator
   - Generate real samples: 
     * Sample n = batch/2 = 16 random x values from [-0.5, 0.5]
     * Compute y = x²
     * Create real labels = 1
   
   - Generate fake samples:
     * Sample z from standard normal distribution (batch/2 samples)
     * Generate X_fake = G(z)
     * Create fake labels = 0
   
   - Update Discriminator:
     * D_loss_real = BCE(D(X_real), real_labels)
     * D_loss_fake = BCE(D(X_fake), fake_labels)
     * D_loss = D_loss_real + D_loss_fake
     * Backpropagate and update D

   // Step 2: Train Generator
   - Sample z from standard normal distribution (batch size samples)
   - Create target labels = 1 (fool discriminator into thinking fake is real)
   - Generate X_fake = G(z)
   - G_loss = BCE(D(X_fake), target_labels)
   - Backpropagate and update G

   // Step 3: Periodic Evaluation
   - At specified epochs [0,1000,2000,...,15000]:
     * Compute real accuracy: D correctly classifies real points as real
     * Compute fake accuracy: D correctly classifies fake points as fake
     * Store generated samples for visualization

3. OUTPUT:
   - Trained Generator that produces points approximating y = x²
   - Training history (losses, accuracies)
   - Visualization of generated samples at different training stages