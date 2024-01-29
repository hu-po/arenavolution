# Generative Model Evolution

<!-- ![o](docs/cover.jpeg) -->

Can we evolve the best vision models? Seed with human examples, "reproduce" them to create novel variants using an LLM, and then benchmark them against a random synthetic classification dataset created using SDXL.

### PyTorch

```
pytorch/
  traineval.py
  test.sh
  models/
    vit.py
    cnn.py
```
traineval.py
test.sh

### TinyGrad?

Could we evolve and beat a standing benchmark for a vision model?

https://github.com/tinygrad/tinygrad/blob/67a78615e5425faad261435f2a78bbc404769a1f/test/Dockerfile
https://github.com/tinygrad/tinygrad/tree/67a78615e5425faad261435f2a78bbc404769a1f/extra/models
https://github.com/tinygrad/tinygrad/blob/67a78615e5425faad261435f2a78bbc404769a1f/examples/train_resnet.py

## Citation

```
@misc{generative-model-evolution-2024,
  title={Generative Model Evolution},
  author={Hugo Ponte},
  year={2024},
  url={https://github.com/hu-po/gme}
}
```