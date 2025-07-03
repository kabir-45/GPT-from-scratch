# LLM from Scratch

This repository contains an implementation of a Transformer model from scratch, inspired by the GPT architecture. The model is trained on a text dataset and includes functionalities for tokenization, training, and text generation. 

---

## ✨ Features

- **Custom Tokenizer**: Implements tokenization using byte pair encoding (BPE) via the `tiktoken` library.  
- **Transformer Architecture**: Includes multi-head attention, feed-forward layers, and layer normalization.  
- **Training Pipeline**: Supports training with customizable hyperparameters, including batch size, context length, and learning rate.  
- **Text Generation**: Generates text using temperature scaling and top-k sampling for diverse outputs.  
- **Pre-trained Weights**: Option to load pre-trained GPT-2 weights for fine-tuning or inference.

---

## 📦 Requirements

- Python 3.7+  
- PyTorch 2.5.1+  
- `tiktoken`  
- `requests`  
- `numpy`  

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/transformer-from-scratch.git
cd transformer-from-scratch
```
---

## ⚙️ Configuration

The model configuration is defined using a dictionary `cfg`. You can modify this to adjust model size, depth, and training behavior:

```python
cfg = {
    "vocab_size": 50257,        # Size of the tokenizer vocabulary
    "context_length": 256,      # Maximum sequence length the model sees
    "emb_dim": 768,             # Dimension of the token embeddings
    "n_layer": 12,              # Number of transformer blocks (layers)
    "n_head": 12,               # Number of attention heads per block
    "dropout": 0.1,             # Dropout rate for regularization
    "qkv_bias": False           # Whether to use bias in QKV projections
}
```
---

## 📄 License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by the ["LLMs from Scratch"](https://github.com/karpathy/ng-video-lecture) tutorial.  
- Uses the `tiktoken` library for tokenization.  
- Pre-trained weights sourced from OpenAI's GPT-2.
