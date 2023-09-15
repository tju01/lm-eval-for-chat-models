## Overview

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

## Implementation

Prompt templates could be added in the following way:
1. Add an option to specify the prompt template in the `--model_args` flag.
2. Implement the code that maps the conversation to a prompt string or list of tokens. There are a some existing libraries, but I hadn't found one that is suitable for evaluation due to insufficient stability, correctness and system message support. The current implementations from FastEval could be moved into a separate library, but I will also look again to see if there is a good existing library for this by now.
3. Add one or multiple functions to the `LM` class for chat inference. The input to those methods would be a conversation instead of a prompt string. This is because this chat functionality can also work for other models like API models that directly use the conversation.
4. Use the templates in tasks. While it would be possible to add an option for using prompt templates for every task, I'm not sure this provides much value. The template is most important for tasks that are specifically designed for chat language models like MT-Bench.
