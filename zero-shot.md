# Larger focus on zero-shot

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
