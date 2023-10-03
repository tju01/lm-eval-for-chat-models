# 1. Proposal for inference modifications

The inference code in lm-evaluation-harness needs to be modified to deal with multi-turn conversations and integrate faster inference methods like vLLM that are very important for longer text generation tasks.

## 1.1. Worker processes

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
1. **Works for vLLM (+ others)**: This approach works very well with other inference methods like vLLM or text-generation-inference that are both used. Both can easily be used in combination with data parallel evaluation which is sometimes important for many GPUs & small models since the default approach of distributing the model across all GPUs [is not good](img/why-data-parallel-is-required.png) ([source](https://discord.com/channels/981279233835958313/1022160802632966186/1151930065576263700)).
2. **Enables data parallel for larger HF transformer models**: If HF transformers needs to be used instead of vLLM for some reason, then this approach still works even for 70B models that need to be split across GPUs. If someone has 8 GPUs, it's very easy to assign every worker 2 GPUs and run the model 4 times in data parallel mode.
3. **Can be adapted easily for asynchronous multi-turn inference**

## 1.2. Asynchronous multi-turn

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

# 2. Specific implementation

## 2.1. Current inference architecture of lm-evaluation-harness

The following places depend on the current inference architecture based on `torch.distributed` and would need to be modified:
1. `evaluator.py` is the main location where the inference happens right now
2. `models/huggingface.py` contains some code dependent on the current multi-GPU architecture there
3. `task.py` has some code for collecting data that is dependent on the rank
4. Some places use a progress bar that depends on the rank

## 2.2. Important considerations

### 2.2.1. Model as argument instead of model path

In lm-evaluation-harness, it is possible to call the [`evaluate` method](https://github.com/EleutherAI/lm-evaluation-harness/blob/f2e3950be5686ff7d3c8c955fb7783a799ed5572/lm_eval/evaluator.py#L153) method directly with a `lm` python object.
FastEval does not have an option to do this and only allows passing a model path.

The problem with this feature is that in a worker-based architecture, the workers are separate child processes and these child processes load the models.
Passing a separate model from the outside isn't really possible.

I'm not sure how important this feature actually is and whether it would be acceptable if this feature would be removed while modifying the inference architecture.
I think the only case where this feature is useful is if the model should be evaluated during training (and is therefore already loaded), but I'm not sure what other tools actually makes use of this.

### 2.2.2. FSDP

Since lm-evaluation-harness is based on `torch.distributed`, it also supports FSDP which FastEval does not.

It's not quite clear how important FSDP is for lm-evaluation-harness.
Moving completely to a worker-based architecture like in FastEval would remove FSDP support.
It would be possible to keep/add FSDP support in the workers at the cost of additional complexity.
But it's not quite clear to me whether it's actually important, so it would be good to discuss that.
