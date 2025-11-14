from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

        # Ensure padding works for batched generation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        You don't need to change this function for now.
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        gen_kwargs = dict(
            max_new_tokens=32,  # enough for <answer>...</answer> + number
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=temperature > 0,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = float(temperature)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        new_tokens = outputs[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Batched version of `generate` method.

        You will likely get an up to 10x speedup using batched decoding.

        To implement batch decoding you will need to:
        - tokenize the prompts self.tokenizer with padding=True and return_tensors="pt"
        - call self.model.generate
        - decode the outputs with self.tokenizer.batch_decode

        Tip: You need to set self.tokenizer.padding_side = "left" to get the correct padding behavior for generation.
             Left padding makes sure all sequences are aligned to the right (i.e. where tokens are generated).
        Tip: self.model.generate takes a lot of parameters. Here are some relevant ones:
            - max_new_tokens: The maximum number of tokens to generate. Set this to a reasonable value
                              (50 should suffice).
            - do_sample and temperature: For any temperature > 0, set do_sample=True.
                                         do_sample=False will use greedy decoding.
            - num_return_sequences: The number of sequences to return. Note that this will generate a flat
                                    list of len(prompts) * num_return_sequences entries.
            - eos_token_id: The end of sequence token id. This is used to stop generation. Set this
                            to self.tokenizer.eos_token_id.
        Pro Tip: Only batch_decode generated tokens by masking out the inputs with
                 outputs[:, len(inputs["input_ids"][0]) :]
        """
        enc = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,  # keep raw question format
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        gen_kwargs = dict(
            max_new_tokens=32,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=temperature > 0,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = float(temperature)
        if num_return_sequences is not None and num_return_sequences > 1:
            gen_kwargs["num_return_sequences"] = int(num_return_sequences)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        # Slice per row based on left padding
        input_lengths = attention_mask.sum(dim=1)
        decoded = []
        for i in range(outputs.size(0)):
            cont_i = outputs[i, int(input_lengths[i].item()):]
            decoded.append(self.tokenizer.decode(cont_i, skip_special_tokens=True).strip())

        if num_return_sequences and num_return_sequences > 1:
            return [decoded[i:i+num_return_sequences] for i in range(0, len(decoded), num_return_sequences)]
        return decoded
    
    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        prompts = [self.format_prompt(q) for q in questions]
        out: list[str] = []
        BS = 32  # micro-batch
        for i in range(0, len(prompts), BS):
            chunk = prompts[i:i+BS]
            gens = self.batched_generate(chunk, temperature=0.0)
            out.extend(gens)
        return [self.parse_answer(g) for g in out]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
