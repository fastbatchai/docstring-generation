# Lesson 1: Introduction to LLM Fine-tuning

*Welcome to the first lesson of the AutoDoc course! In this comprehensive introduction, we'll explore the fundamental concepts and techniques for fine-tuning language models on downstream tasks.*


<details>
<summary><strong>📋 Table of Contents</strong></summary>

1. [Instruction Tuning](#instruction-tuning)
   - [Why Instruction Tuning?](#why-instruction-tuning)
   - [The Instruction Tuning Process](#the-instruction-tuning-process)
   - [Key Factors for Generalization](#key-factors-for-generalization)
   - [Chain-of-Thought (CoT) Reasoning Fine-tuning](#chain-of-thought-cot-reasoning-fine-tuning)

2. [Parameter-Efficient Fine-tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
   - [Adapters](#adapters)
   - [Prefix Tuning](#prefix-tuning)
   - [Prompt Tuning](#prompt-tuning)
   - [Low-Rank Adaptation (LoRA)](#low-rank-adaptation-lora)
   - [LoRA Variants](#lora-variants)

3. [Knowledge Distillation](#knowledge-distillation)

4. [Reinforcement Learning for LLMs](#reinforcement-learning-for-llms)
   - [RL from Human Feedback (RLHF)](#rl-from-human-feedback-rlhf)
   - [Direct Preference Optimization (DPO)](#direct-preference-optimization-dpo)
   - [Group Relative Policy Optimization (GRPO)](#group-relative-policy-optimization-grpo)

5. [Test-Time Scaling](#test-time-scaling)

6. [Key Takeaways](#key-takeaways)
7. [What's Next?](#whats-next)
8. [Additional Resources](#additional-resources)

</details>

---

## 🎯 What You'll Learn

- **Core concepts** of LLM fine-tuning and when to use different approaches
- **Instruction tuning** fundamentals and supervised fine-tuning techniques
- **Parameter-efficient methods** like LoRA, QLoRA, and adapters
- **Reinforcement learning** approaches including GRPO and RLHF
- **Other post-training methods** like knowledge distillation and Test-time scaling


🚀 Let's Begin!

## [Instruction Tuning](https://crfm.stanford.edu/2023/03/13/alpaca.html)

Instruction tuning is a **supervised** training process that helps pretrained LLMs generalize to **unseen tasks**. This technique fine-tunes a pre-trained LLM on pairs of instructions and desired outputs, teaching the model to map instructions to desired outcomes that transfer to new tasks.
These instructions and desired outputs are instances or examples on how to solve or answer a specific task (e.g., Q&A, summarization, ..).

### Why Instruction Tuning?

- **Improves task understanding**: Helps LLMs learn patterns for mapping instructions to desired outcomes
- **Enables generalization**: Models trained on diverse instruction sets can handle unseen tasks
- **Enhances controllability**: Makes models more responsive to user intent and formatting requirements

### The Instruction Tuning Process

**Step 1: Collect or Construct Instruction-Formatted Instances**: An instruction-formatted instance consists of:
- **Task description** (the instruction)
- **Optional input** (parameters for the instruction)
- **Desired output** (the expected response)
- **Optional demonstrations** (few-shot examples)

Constructing high-quality instruction datasets traditionally requires human annotation, which can be costly. Semi-supervised approaches leverage existing datasets by using capable LLMs to synthesize diverse task descriptions and generate additional training instances.

**Step 2: Fine-tune LLMs Using Supervised Learning**:
The model is fine-tuned using standard supervised learning with causal language modeling (i.e., predict the next token). Given an instruction-output pair, the objective is to minimize the negative log-likelihod (NLL) on the desired output given the input sequence (instruction + input). During training the instruction and input are masked and only the output tokens are considered in the loss computation.

Since instruction datasets include a mixture of tasks, it's important to balance the proportion of data from each task. Common strategies include:

- **Examples-proportional mixing**: All datasets are combined and instances are sampled uniformly
- **Maximum cap**: Limits the maximum number of examples per dataset to prevent larger datasets from overwhelming the distribution
- **Multi-stage fine-tuning**: Sequential training on different task categories



### Key Factors for Generalization

To achieve generalization on unseen tasks, consider these factors when creating instruction datasets:

- **Scaling**: Increasing the number of tasks improves generalization up to a certain level (performance saturates after a certain number of tasks)
- **Diversity**: Task descriptions should vary in length, structure, and creativity to improve generalization performance
- **Quality over quantity**: Well-crafted, diverse examples often outperform larger datasets with repetitive patterns


### Chain-of-Thought (CoT) Reasoning Fine-tuning

CoT reasoning fine-tuning is a specialized form of instruction fine-tuning where the model is trained to produce **step-by-step reasoning traces** rather than directly to final answers. Instead of just providing the solution in the training instance, the desired output includes intermediate reasoning steps that show how to arrive at the answer.

### 🔧 Parameter-Efficient Fine-tuning (PEFT)

Instead of fine-tuning all the model's parameters (which requires significant memory and computation), PEFT aims to reduce the number of trainable parameters while retaining good performance. This approach is crucial when working with large models or limited computational resources. Below, I will discuss different PEFT methods:


#### [Adapters](https://arxiv.org/abs/1902.00751)

Adapters introduce small modules (small neural networks) after Transformer feedforward layers. Only the adapter modules are trained, while the base model remains frozen. These adapters have a bottleneck structure where first the feature dimension (from Transformer FF) is projected into a smaller dimension followed by a nonlinearity, finally the original dimension is recovered (see figure below)

<div align="center">

![Adapter Architecture](./figures/adapteurs.png "Architecture of the adapter module and its integration with the Transformer ([Source](https://arxiv.org/pdf/1902.00751))")

</div>

#### [Prefix Tuning](https://arxiv.org/abs/2101.00190)
This method is inspired by prompting, where the intuition is that providing an appropriate context can steer a language model’s output toward a desired goal without modifying its internal parameters. Traditional prompting uses natural language instructions or examples appended to the input text to guide the model’s behavior.

Prefix tuning extends this idea by introducing a sequence of prefix indices that reserve positions at the beginning of the input. These indices do not directly contain vectors but indicate where learnable prefix embeddings will be inserted within the model. During computation, each prefix index $𝑖 \in 𝑃_{idx}$
is replaced by its corresponding trainable vector $𝑃_\theta[i,:]$ in the hidden activations. This ensures that all subsequent tokens are conditioned on the learned prefix representations. The collection of prefix parameters is stored in a matrix $P_\theta$ which is optimized through backpropagation while the base language model parameters remain frozen.

<div align="center">

![Prefix_tuning](./figures/prefix_tuning.png "Illustration of prefix tuning ([Source](https://arxiv.org/pdf/2101.00190))")

</div>

#### [Prompt Tuning](https://arxiv.org/pdf/2104.08691)

Prompt tuning is a a simplified version of prefix tuning that learns **soft prompts** (prompt embeddings) at the input layer **only**. Soft prompts are learnable tokens that are appended to the input tokens. These tokens are updated using backpropagation while keep the model frozen. This method is more effective with large models.

#### [Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685)

LoRA is one of the most popular PEFT methods. It approximates the weight update matrices $ΔW$ with a low-rank decomposition: $ΔW = A.B^T$. Only matrices $A$ and $B$ are trainable for task adaptation. Once these matrices are learnt, they can be merged with the base model $W^{'} = W + A.B^T$.

In the original paper, only the attention projection weights (i.e., Query, Key, Value and Output projections) ara adapted while freezing the base model.

#### LoRA Variants

- **[QLoRA](https://arxiv.org/abs/2305.14314)**: Combines LoRA with 4-bit quantization (i.e., the base model is loaded in 4-bit quantization)
- **[Rank-Stabilized LoRA](https://arxiv.org/abs/2312.03732)**: Normalizes/scales the adapter updates, making training less sensitive to the rank parameter
- **[Decomposed LoRA](https://arxiv.org/pdf/2402.09353)**: Decomposes pretrained weights into magnitude and directional components, fine-tuning both. The directional component is fine-tuned using LoRA.

## [Knowledge Distillation](https://arxiv.org/abs/1503.02531)
Knowledge distillation is a well-known model compression technique in deep learning. It involves using larger, more capable models (teachers) to produce training data or guide the training of smaller models (students). This approach can result in lighter, faster models with performance approaching that of the larger model. This approach helps reduce inference costs and latency.

[Chain of Thought Distillation](https://arxiv.org/abs/2212.08410) is an example where a powerful LLM generates intermediate reasoning steps for complex problems, and the student LLM is fine-tuned to reproduce both the reasoning chain and final answer. This transfers not just knowledge but also reasoning capabilities.

## Reinforcement Learning for LLMs
Since LLMs generate each token prediction based on the previously generated tokens, this process can be modelled as a **Markov Decision Process** (MDP) as follows:
- **state**: the generated tokens so far
- **action**: the next token
- **reward function**: evaluate the quality of the output based on action and state
- **Policy**: the model's output distribution over the next token = The LLM itself

<div align="center">

![RLHF](./figures/RLHF.png "RLHF pipeline ([Source](https://arxiv.org/abs/2203.02155)")

</div>


MDP is the mathematical framework underlying all RL algorithms. An MDP models sequential decision-making where an agent interacts with an environment to maximize cumulative reward.

An LLM policy can be learned by maximixing the expected discounted returns. RL for LLM is usually used to align the model with human preferences and the process follows these steps:
1) **Supervised Fine-Tuning (SFT)**: starts with a pretrained language model that is subsequently refined on a supervised dataset of high-quality, human-crafted examples. This phase ensures the model acquires a baseline compliance with format and style guidelines.
2) **Reward Model (RM) Training**: Collect generated outputs from the fine-tuned model and obtain human preference labels (rankings or scores). Train a reward model to replicate these preference judgments, learning a continuous reward function that maps response text to a scalar value.
3) **RL Fine-Tuning**: Optimize the language model using a policy gradient algorithm (e.g., PPO) to maximize the reward model's output. Through iterative training, the LLM learns to produce responses that humans find preferable along key dimensions such as accuracy, helpfulness, safety, and stylistic coherence.


### [RL from Human Feedback (RLHF)](https://arxiv.org/abs/2203.02155)
RLHF is the foundational approach for aligning LLMs with human preferences using the three steps explained above. The reward model is trained using  human-annotated **rankings** of generated responses and [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) is the policy gradient algorithm used to finetune the LLM.

**[RLAIF (RL from AI Feedback)](https://arxiv.org/abs/2309.00267)** is an alternative to RLHF that replaces the human annotations with AI generated ones. RLAIF uses a second highly capable LLM to generate the preference labels used to train the reward model. This helps eliminate the cost of human annotation while maintaining a good performance and alignment.

### [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)
Instead of using an multi-stage RL pipeline (learning a reward model and running policy-gradient updates), DPO directly integrated the human preferences in the LLM training objective. The key idea is to increase the likelihood of the preferred response while decreasing the likelihood of the less-preferred response.
Compared to RLHF, DPO simplifies the training pipeline, however human preferences are still needed to train DPO.
### [Group Relative Policy Optimization (GRPO)](https://arxiv.org/abs/2402.03300)
GRPO was introduced in DeepSeek-R1 models.  GRPO simplifies the advantage estimation process. PPO uses avantage function $A(s,a) = Q(s,a) - V(s)$ to quantify how much better an action $a$ is than the basline expected return $V(s)$. GRPO doesnt use a sperate value function as baseline V, it estimates $V(s)$ as the average reward of multiple sampled outputs for the same prompt.

For each prompt $p$, GRPO samples a **group** of outputs ${o_1, o_2, . . . , o_G}$ from the current policy $\pi_\theta$ (i.e., the LLM). A reward model is used to score each output in the group, yielding rewards ${r_1, r_2, . . . , r_G}$. The group rewards are normalized by subtracting the group average and dividing by the standard deviation.

GRPO can be applied in two modes:
- **Outcome Supervision**: Provides a reward only at the end of each output. The advantage $A_{i,t}$ for all tokens in the output is set as the normalized reward.
- **Process supervision**:  Provides rewards at the end of each reasoning step, allowing more fine-grained credit assignment and potentially better learning of complex reasoning patterns.

<div align="center">

![Comparison between PPO, GRPO and DPO](./figures/comaprison_ppo_dpo_grpo.png "Comparison between PPO, GRPO and DPO([Source](https://arxiv.org/abs/2402.03300))")

</div>


### ⚡ Test-Time Scaling

Test-time scaling refers to techniques that improve model performance during inference without modifying the core architecture. This includes:

- **[Chain-of-thought prompting](https://arxiv.org/abs/2201.11903)**: Encouraging step-by-step reasoning
- **[Self-consistency](https://arxiv.org/abs/2203.11171)**: Sampling multiple reasoning paths and selecting the most consistent answer
- **[Iterative refinement](https://arxiv.org/abs/2303.17651)**: Having the model critique and improve its own outputs


These approaches improve generalization across tasks but introduce additional computational costs at inference time.


## 🚀 What's Next?

Now that you understand the theoretical foundations, you're ready to dive into practical implementation:

- **Lesson 2**: Data preparation for multi-language docstring generation
- **Lesson 3**: Building instruction fine-tuning pipelines with Modal
- **Lesson 4**: Implementing GRPO for preference-based learning
- **Lesson 5**: Comprehensive evaluation and model comparison

## 📚 Additional Resources

- [LLM Post-Training: A Deep Dive into Reasoning
Large Language Models](https://arxiv.org/pdf/2502.21321)
- [Understanding Prompt Tuning: Enhance Your Language Models with Precision](https://www.datacamp.com/tutorial/understanding-prompt-tuning)
- [Mastering QLoRa](https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html)
---

*Ready to start building? Let's move on to Lesson 2: Data Preparation!*
