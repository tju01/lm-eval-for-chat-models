# 1. More flexible tasks

## 1.1. Overview

The current `Task` class is designed for tasks that have some test dataset, possible some few-shot context and have only a single turn.
This structure is insufficient for benchmarks like MT-Bench that require a lot more flexibility.
The only assumptions I would make about a task are
1. It has an `evaluate` method.
2. This method runs inference with the language model repeatedly.
4. It returns a score and possibly other information.
5. It can give intermediate progress updates that can be used for a progress bar.

## 1.2. MT-Bench

This benchmark consists of 80 conversations with 2 turns each.
For every turn, the model is queried to give an answer.
The second turn input depends on the first turn output.
In the end, all 160 model responses are sent to GPT-4 for review which then returns scores between 1 and 10.
The average score over all 160 turns is the final MT-Bench score.

While the number of questions is very low, they seem to be carefully selected.
The result seems to be one of the best benchmarks for evaluating chat language models.

## 1.3. CoT

CoT reasoning tasks like GSM8K and MATH can be implemented relatively easily using the current `Task` class.
I also saw that there is already an existing effort to add GSM8K in the few-shot setting.
For MATH, it seems like context length is a problem for the few-shot CoT case as well as the lack of scores in the original paper for this setting.
After adding support for prompt templates, the zero-shot case for chat language models should be possible for both tasks.
The main question for zero-shot is how to prompt the model & extract the answer.

## 1.4. HumanEval

WizardCoder also used their prompt template for HumanEval and I also implemented this prompted version.
However, this improvement would be better implemented in bigcode-evaluation-harness which currently supports a very limited and hardcoded version of these prompt templates.

## 1.5. Agent Benchmarks

I have started working on agent benchmarks like [AgentBench](https://github.com/THUDM/AgentBench) in FastEval.
While I haven't had success with it yet due to being unable to run their full code & reproduce their results, I & others are generally interested in having these kind of tasks.
Ideally it would be possible to also integrate them into lm-evaluation-harness afterwards, even if that doesn't happen in the beginning.
But it would be good to design the general architecture with it in mind.
