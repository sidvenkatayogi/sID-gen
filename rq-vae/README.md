## RQ-VAE modules

ref: [RQ-VAE](https://arxiv.org/abs/2203.01941)

## some notes:
- this was meant to be just simple rq-vae but i jumped a little ahead and ended up just adding some TIGER design into this
- one codebook for each level (not shared)
- fixed codebook size across levels
- initialize codebook with k means cluster vectors
- re initialize not used / dead codes with random vector
- codebook is updated using exponential moving average of assigned residuals
- decoder module not used after training (goal is to create quality semantic IDs, not reconstruction)
- first trained and tested on MovieLens-1M