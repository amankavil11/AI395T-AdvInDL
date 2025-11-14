from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        messages = [
            {
                "role": "system",
                "content": "You are a precise unit-conversion calculator. Be concise. Always end with <answer>NUMBER</answer>",
            },
            {
                "role": "user",
                "content": "How many m is 3 km?",
            },
            {
                "role": "assistant",
                "content": "1 km = 1000 m. 3 times 1000 = <answer>3000</answer>",
            },
            {
                "role": "user",
                "content": "How many MB is 2 G?",
            },
            {
                "role": "assistant",
                "content": "1 G = 1000 MB. 2 times 1000 = <answer>2000</answer>",
            },
            {
                "role": "user",
                "content": question,
            },
        ]
        # Use the tokenizer's chat template so the instruct model behaves correctly.
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )



def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
