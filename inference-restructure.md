# Current inference architecture of LM-Eval

Quite dependent on `torch.distributed`.
Some places that would need to be changed to have an asynchronous worker-based approach instead:
- `evaluator.py`
  - This is the main location where the inference happens right now
- `models/huggingface.py`
  - There is some code dependent on the current multi-GPU architecture there
- `task.py`
  - There is code for collecting data on each rank there
- Some places that use a progress bar which depends on the rank

# Important things to consider

## Giving model as argument instead of model path

The inference architecture from fasteval doesn't really work directly like that.
It needs to be modified so that a model can be given as an argument.
FastEval doesn't support that, it only supports giving the model path and it wants to load the model in a worker process by itself.
But if the model is given as an argument, I don't think it can be moved to a worker process...
So how to deal with that?

## FSDP

Not sure how much it matters that this still works afterwards.

# Some arguments for modifying inference

![image](https://github.com/tju01/lm-eval-for-chat-models/assets/70238802/ba0872dd-7366-4af3-9cb0-0aec73bcdf8f)

Stolen from https://discord.com/channels/981279233835958313/1022160802632966186/1151930065576263700
