# 4. Inference modifications

The inference code in lm-evaluation-harness needs to be modified to deal with multi-turn conversations and integrate faster inference methods like vLLM that are very important for longer text generation tasks.

## 4.1 Worker processes

Currently, the inference in lm-evaluation-harness is quite integrated with `torch.distributed`.
Inside the `evaluate` function, there is code conditional on `lm.rank` for collecting the requests in the beginning, executing them on each rank and then collecting them in the end again.
This works reasonably well for using HF transformers with `torch.distributed` in a single conversation turn, but it seems hard to use for more complex requirements.

I'm writing down some notes here about how this is handled in FastEval.
That is not to say that the structure should be changed like this.
It's just that I'm quite unsure about how the benchmarks & faster inference could work well with the current inference structure and I would like to discuss this aspect further.

In FastEval, instead of using `torch.distributed`, there is only a single main process and the main code only runs once.
There are a number of worker processes, but the _only_ task of these workers is to do inference and nothing else.
Communication between the main process and the workers happens through `multiprocessing.Queue`.
This approach keeps the main code simple since it only runs once and not for every rank.
In addition, it also has the following benefits that I'm not sure how to make work with the current inference architecture based on `torch.distributed`:
1. **Works for vLLM (+ others)**: This approach works very well with other inference methods like vLLM or text-generation-inference that are both used. Both can easily be used in combination with data parallel evaluation which is sometimes important for many GPUs & small models since the default approach of distributing the model across all GPUs is not the most optimal one here.
2. **Enables data parallel for larger HF transformer models**: If HF transformers needs to be used instead of vLLM for some reason, then this approach still works even for 70B models that need to be split across GPUs. If someone has 8 GPUs, it's very easy to assign every worker 2 GPUs and run the model 4 times in data parallel mode.
3. **Can be adapted easily for asynchronous multi-turn inference**

## 4.2 Asynchronous multi-turn

Multi-turn conversations that are required by benchmarks like MT-Bench can either be handled in a synchronous or in an asynchronous way.

**Synchronous**:
The synchronous method would be to repeatedly collect all inference requests for a turn, execute them, send the results back to the tasks and continue until all tasks have reached the last turn.
The main problem with this method is performance.
At every turn, we would need to wait until the last inference request has finished before starting the next turn.
This is not optimal for batching since only a small number of requests will remain in the batch towards the end of a turn and it would be better to add inference requests from the next turn to the batch.

**Asynchronous**:
An asynchronous approach solves this problem.
Once the first turn of a conversation ended, we can immediately continue with the second turn and add the corresponding request to the batch.
This will always keep the GPUs as busy as possible right until the end of all turns.

Note that while the difference might not be _that_ large for two turns as in MT-Bench, the difference will be a lot larger once many turns are involved.
I am currently working on agent benchmarks in FastEval that involve an interactive environment where additionally we don't want to run all of the first turns at once.
Instead, we only deal with a smaller number of conversations (environment trajectories) at once which will make this problem even worse if a synchronous architecture is used.
For these reasons, I would highly recommend the asynchronous approach and it's also what I use in FastEval using `asyncio`.

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

![image](https://github.com/tju01/lm-eval-for-chat-models/assets/70238802/137c5fe0-0fe4-42c6-b73d-86f5bdb07b5a)

Stolen from https://discord.com/channels/981279233835958313/1022160802632966186/1151930065576263700
