<div align="center">
<h1> AutoDoc Course</h1>
<h3>Learn how to finetune language models to generate docstrings</h3>
<p class="tagline">Open-source course by <a href="https://substack.com/@fastbatch">Fast Batch</a></p>
</div>



**AutoDoc** is short and comprehensive implemntation of an LLM finetuning pipeline to automatically generate high-quality docstrings for code functions across multiple programming languages. This repo includes training and evaluation scripts for `instruction finetuning` and `RL-finetuning` using `GRPO`. It also helps compare different finetuning frameworks such as [PEFT](https://huggingface.co/docs/peft/index), [Unsloth](https://unsloth.ai/) and [TRL](https://github.com/huggingface/trl?tab=readme-ov-file).

We use [Modal](https://modal.com/) for training and evaluating on GPU. It possible to finetune a 2B parameter model almost for free using Modal. As a reference, all our experiments cost us 30$. 

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/fastbatchai/docstring-generation.git
cd docstring-generation

# Install dependencies
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```
You need to setup Modal before starting training ```modal setup```

### Launch instruction finetuning
```bash
modal run -i -m autoDoc.train --training-type sft --use-unsloth
```
### Launch RL-finetuning using GRPO
```bash
modal run -i -m autoDoc.train --training-type grpo --experiment_name EXPERIMENT_NAME --verbose 1
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


<!-- **Made with ❤️ for the open-source community** -->