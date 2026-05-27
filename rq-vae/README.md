## RQ-VAE modules

## some notes:
- one codebook for each level (not shared)
- fixed codebook size across levels
- initialize codebook with k means cluster vectors
- re initialize not used / dead codes with random vector
- codebook is updated using exponential moving average of assigned residuals
- decoder module not used after training (goal is to create quality semantic IDs, not reconstruction)
- first trained and tested on MovieLens-1M