# Introduction

This document describes a proposal for changes in lm-evaluation-harness to integrate benchmarks for instruction-finetuned/chat language models.
While lm-evaluation-harness is often used for these models, the currently implemented benchmarks are primarily designed around base models.

I have implemented all the ideas here in [FastEval](https://github.com/FastEval/FastEval) and could contribute significantly to integrate them in lm-evaluation-harness.
However, it would require large changes and it would be good to discuss these things in more detail first.

# Overview

To show what changes would be required, let's look at some of the benchmarks for chat language models first.
Here is a list of the benchmarks that are used by the [WizardLM](https://github.com/nlpxucan/WizardLM) group for evaluating their (very good) instruction-following/finetuned models that are based on existing base models like LLaMA:
1. MT-Bench	& AlpacaEval for conversational abilities
2. GSM8K & MATH in a zero-shot CoT setting for mathematical reasoning abilities
3. HumanEval & MBPP for Python coding abilities

From this, we can identify the following aspects that are currently missing from lm-evaluation-harness:
1. **Prompt templates**: All of their benchmarks prompt the language model using the corresponding [prompt template](https://github.com/FastEval/FastEval/blob/main/docs/model-type.md) that their model was fine-tuned with. This makes the evaluation closer to how the models will be used in practice.
2. **Larger focus on zero-shot**: None of these benchmarks use a few-shot context. All of them are evaluated using a zero-shot setting. Again, this is also how the models are mostly used later in practice.
3. **More flexible tasks**: MT-Bench in particular is very different from the existing tasks in lm-evaluation-harness. In addition to the zero-shot prompted setting, it also uses multi-turn conversations and is model-graded by GPT-4. Integrating this benchmark requires supporting a more flexible task structure.
4. **Inference modifications**: Most of the current tasks in lm-evaluation-harness only consider a few output tokens per prompt. In contrast, these benchmarks generate a few hundred tokens. Running these benchmarks using HF transformers is very slow and the WizardLM group has also integrated vLLM in their evaluations.

# 1. Prompt templates

## 1.1 Overview

Chat language models are fine-tuned with a [prompt template](https://github.com/FastEval/FastEval/blob/main/docs/model-type.md).
This does not refer to the kind of prompt templates that can be found in a library like [PromptSource](https://github.com/bigscience-workshop/promptsource).
Instead, it refers to the way that a conversation between a user and an assistant is transformed to a prompt string.
As an example, consider the following conversation:

```json
[
  { "role": "system", "content": "You are a helpful assistant." },
  { "role": "user", "content": "Can you give me some lorem ipsum?" },
  { "role": "assistant", "content": "Sure, here you go!\n\nLorem ipsum dolor sit amet [...]" },
  { "role": "user", "content": "Thanks! Some more please, it's not enough." },
]
```

Using the ChatML prompt template, this conversation would be transformed into the following prompt string that would then be the input to the fine-tuned language model.

```
<|im_start>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Can you give me some lorem ipsum?<|im_end|>
<|im_start|>assistant
Sure, here you go!

Lorem ipsum dolor sit amet [...]<|im_end|>
<|im_start|>user
Thanks! Some more please, it's not enough.<|im_end|>
```

While the vast majority of chat models use a prompt template that just involves some simple string concatenation like this, **[some models](https://huggingface.co/openchat/openchat#conversation-template) require working on a token level** due to the `tokenize(A) + tokenize(B) != tokenize(A + B)` problem.
A correct implementation also needs to take this into account.

## 1.2 Implementation

Prompt templates could be added in the following way:
1. Add an option to specify the prompt template in the `--model_args` flag.
2. Implement the code that maps the conversation to a prompt string or list of tokens. There are a some existing libraries, but I hadn't found one that is suitable for evaluation due to insufficient stability, correctness and system message support. The current implementations from FastEval could be moved into a separate library, but I will also look again to see if there is a good existing library for this by now.
3. Add one or multiple functions to the `LM` class for chat inference. The input to those methods would be a conversation instead of a prompt string. This is because this chat functionality can also work for other models like API models that directly use the conversation.
4. Use the templates in tasks. While it would be possible to add an option for using prompt templates for every task, I'm not sure this provides much value. The template is most important for tasks that are specifically designed for chat language models like MT-Bench.

# 2. Larger focus on zero-shot

Chat language models are mostly used in a zero-shot setting in practice.
It is therefore important to have benchmarks that measure these capabilities.
While lm-evaluation-harness can already be used this way, some important tasks make strong assumptions about the exact way the model will respond and will not work well without a few-shot prompt.

The WizardLM models are trained to respond in a specific way and this is therefore not a problem for them.
However, this approach is not a general solution for all models.
For a more general alternative to few-shot prompting, it is possible to prompt chat models to respond in a certain way like 
`Solve this problem step-by-step and finally answer with the final solution in the last line`.
Additional answer extraction code also helps.

I have used this approach for the CoT tasks in FastEval.
This means that the prompting & answer extraction code is different than for WizardLM which also leads to different scores.
The test data & answer checking is still exactly the same and also the same as in lm-evaluation-harness.
I considered this to be the best solution, but I'm not sure what the policy is for including these things in lm-evaluation-harness.

This is not a problem for tasks like MT-Bench that are designed for zero-shot in the first place.

# 3. More flexible tasks

## 3.1 Overview

The current `Task` class is designed for tasks that have some test dataset, possible some few-shot context and have only a single turn.
This structure is insufficient for benchmarks like MT-Bench that require a lot more flexibility.
The only assumptions I would make about a task are
1. It has an `evaluate` method.
2. This method runs inference with the language model repeatedly.
4. It returns a score and possibly other information.
5. It can give intermediate progress updates that can be used for a progress bar.

## 3.2 MT-Bench

This benchmark consists of 80 conversations with 2 turns each.
For every turn, the model is queried to give an answer.
The second turn input depends on the first turn output.
In the end, all 160 model responses are sent to GPT-4 for review which then returns scores between 1 and 10.
The average score over all 160 turns is the final MT-Bench score.

While the number of questions is very low, they seem to be carefully selected.
The result seems to be one of the best benchmarks for evaluating chat language models.

## 3.3 CoT

CoT reasoning tasks like GSM8K and MATH can be implemented relatively easily using the current `Task` class.
I also saw that there is already an existing effort to add GSM8K in the few-shot setting.
For MATH, it seems like context length is a problem for the few-shot CoT case as well as the lack of scores in the original paper for this setting.
After adding support for prompt templates, the zero-shot case for chat language models should be possible for both tasks.
The main question for zero-shot is how to prompt the model & extract the answer.

## 3.4 HumanEval

WizardCoder also used their prompt template for HumanEval and I also implemented this prompted version.
However, this improvement would be better implemented in bigcode-evaluation-harness which currently supports a very limited and hardcoded version of these prompt templates.

## 3.5 Agent Benchmarks

I have started working on agent benchmarks like [AgentBench](https://github.com/THUDM/AgentBench) in FastEval.
While I haven't had success with it yet due to being unable to run their full code & reproduce their results, I & others are generally interested in having these kind of tasks.
Ideally it would be possible to also integrate them into lm-evaluation-harness afterwards, even if that doesn't happen in the beginning.
But it would be good to design the general architecture with it in mind.

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
