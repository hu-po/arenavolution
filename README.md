# Eigenops Evolution

Can we evolve the best vision block?

### Local SDXL Dataset

To evaluate the vision blocks they will be given a fixed number of epochs to train on a synthetic dataset.
The synthetic dataset is created by using SDXL with categories picked by GPT-4.
To run SDXL locally use:

```
https://replicate.com/stability-ai/sdxl
docker run --name sdxl \
 -v /home/oop/dev/data/sdxl/sdxl-cache:/path/inside/container/sdxl-cache \
 -v /home/oop/dev/data/sdxl/safety-cache:/path/inside/container/safety-cache \
 r8.im/stability-ai/sdxl@sha256:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b
docker commit sdxl sdxl
```