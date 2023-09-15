This repository describes a proposal for changes in lm-evaluation-harness to integrate benchmarks for instruction-finetuned/chat language models.
While lm-evaluation-harness is often used for these models, the currently implemented benchmarks are primarily designed around base models.

To show what changes would be required, let's look at some of the benchmarks for chat language models first.
The [WizardLM](https://github.com/nlpxucan/WizardLM) group uses the following benchmarks for evaluating their instruction-finetuned/chat language models:
1. MT-Bench	& AlpacaEval for conversational abilities
2. GSM8K & MATH in a zero-shot CoT setting for mathematical reasoning abilities
3. HumanEval & MBPP for Python coding abilities

From this, we can identify the following aspects that are currently missing from lm-evaluation-harness:
1. [**Chat templates**](chat-templates.md): All of their benchmarks prompt the language model using the corresponding chat template that their model was fine-tuned with. This makes the evaluation closer to how the models will be used in practice.
2. [**Larger focus on zero-shot**](zero-shot.md): None of these benchmarks use a few-shot context. All of them are evaluated using a zero-shot setting. Again, this is also how the models are mostly used later in practice.
3. [**More flexible tasks**](more-flexible-tasks.md): MT-Bench in particular is very different from the existing tasks in lm-evaluation-harness. In addition to the zero-shot setting with chat templates, it also uses multi-turn conversations and is model-graded by GPT-4. Integrating this benchmark requires supporting a more flexible task structure.
4. [**Inference modifications**](inference-modifications.md): Most of the current tasks in lm-evaluation-harness only consider a few output tokens per prompt. In contrast, these benchmarks generate a few hundred tokens. Running these benchmarks using HF transformers is very slow and the WizardLM group has also integrated vLLM in their evaluations.
