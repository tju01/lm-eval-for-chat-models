# 1. Overview

Chat language models are fine-tuned with a [chat template](https://github.com/FastEval/FastEval/blob/main/docs/model-type.md) to handle conversations between a user and an assistant.
As an example, consider the following conversation:

```json
[
  { "role": "system", "content": "You are a helpful assistant." },
  { "role": "user", "content": "Can you give me some lorem ipsum?" },
  { "role": "assistant", "content": "Sure, here you go!\n\nLorem ipsum dolor sit amet [...]" },
  { "role": "user", "content": "Thanks! Some more please, it's not enough." },
]
```

Using the ChatML chat template, this conversation would be transformed into the following prompt string that would then be the input to the fine-tuned language model.

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

# 2. Implementation

## 2.1. Option for user to specify the chat template

For now, I wouldn't determine the chat template that should be used for a model automatically.
Instead, the user should specify the template manually.
The best place for doing this is probably as part of the `--model_args` flag.
So it would look something like `--model_args pretrained=...,chat_template=chatml` or so.

## 2.2. Implementing the chat templates

The implementations of chat templates can be pretty isolated from the rest of the code and it's also possible to implement it in an external library with only a very small interface.
The main thing that is required is basically a `apply_chat_template(chat)` method that takes in the chat conversation and outputs the prompt for the model.

One important aspect that is sometimes ignored is that **[some models](https://huggingface.co/openchat/openchat#conversation-template) require working on a token level** due to the `tokenize(A) + tokenize(B) != tokenize(A + B)` problem.
So instead of returning a prompt _string_, a 100% correct implementation of the template would return tokens instead.

### 2.2.1. Using the new HF transformers implementation

HuggingFace has [recently added a feature in their library](https://huggingface.co/docs/transformers/main/en/chat_templating).
This allows the creators of models to specify the template as part of the tokenizer.
_If_ the model creator has specified the template this way, then it can be easily used:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

tokenizer.use_default_system_prompt = False
tokenizer.apply_chat_template(chat, tokenize=False)
```

However, right now this feature is very new and I've not found a model that makes use of this.
Even for the `meta-llama/Llama-2-7b-chat-hf`, the model tokenizer itself doesn't currently specify the template and instead HF just uses the default template that HF itself implements and uses based on the _model type_.
Other models like `OpenAssistant/codellama-13b-oasst-sft-v10` and `Open-Orca/OpenOrcaxOpenChat-Preview2-13B` also use the same template right now (I've tried it) which is actually _incorrect_ for those models since the Open-Assistant model uses ChatML while the Open-Orca model uses another custom template.

This might at some point become the best option, but right now it isn't.

### 2.2.2. Using an external library

There have been a few libraries that implement these chat templates.
The most complete implementation is probably part of [FastChat](https://github.com/lm-sys/FastChat/blob/3149253988ee16b0945aa0a381a42a07b8a7829e/fastchat/conversation.py).
This is a reasonable good implementation, however the main problems with it right now are:
1. It's unstable. They sometimes make changes to existing templates.
2. While it contains many implementations, some of them are slightly incorrect.
3. FastChat is too big of an dependency (though that's easy to solve).

There are a some existing libraries, but I hadn't found one that is suitable for evaluation due to insufficient stability, correctness and system message support. The current implementations from FastEval could be moved into a separate library, but I will also look again to see if there is a good existing library for this by now.

### 2.2.3. Implementation in lm-evaluation-harness itself

Implement the code that maps the conversation to a prompt string or list of tokens.

## 2.3. Adding methods for doing chat inference

Add one or multiple functions to the `LM` class for chat inference. The input to those methods would be a conversation instead of a prompt string. This is because this chat functionality can also work for other models like API models that directly use the conversation.

## 2.4. Using the chat templates in tasks

Use the templates in tasks. While it would be possible to add an option for using chat templates for every task, I'm not sure this provides much value. The template is most important for tasks that are specifically designed for chat language models like MT-Bench.
