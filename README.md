# Arena Evolution

### Rough Ideas

Have LLMs compete in an arena as a form of self-play.
The arena consist of text questions with answers.
Two "player" LLMs go head-to-head and try to answer the question.
Three "judge" LLMs pick the best answer.
The best and worse answers are used to push gradients via DPO into the LLMs.
Each LLM is a MoE Mixtral model, each expert and possibly the gating network has a LoRA on it.
The finetuning is done through DPO.
DPO samples are selected by majority vote on other LLMs.
There is a limited pool of LLMs from which players and judges are randomly sampled.
Slower outer loop of evolution: low scoring LLMs are culled, high schoring LLMs "sexually" reproduce by shuffling their expert LoRAs.

### Possible Pitfalls

- Loading each LLM judge and player at each question will be slow locally. Perhaps experience collection can be done with frozen LLMs ideally via API. GPT would be great for this.
- Putting a LORA on the MoE experts is uncertain at best. The codebase for Mixtral may not allow this or make it simple.
- DPO is also likely not implemented in any way relevant to Mixtral. Lots of original code will need to be written.
- Culling and reproduction will be hard to log and thus track.
- How do we eval and see if progress is being made, as opposed to some kind of mode collapse or degenerate behavior?

### Code Concerns

Implement more from scratch for better understanding and practice. OR implement on top of existing implementations frankenstein style for maximum build speed. What do I want out of this project?

MoE training on top of Megatron by T Gale
https://github.com/stanford-futuredata/megablocks/tree/main

Mistral has their own fork of Megatron
https://github.com/mistralai/megablocks-public

DPO implementation by paper authors
https://github.com/eric-mitchell/direct-preference-optimization?tab=readme-ov-file

Mistral implementation by Mistral team
https://github.com/mistralai/mistral-src