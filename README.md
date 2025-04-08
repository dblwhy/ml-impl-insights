# ML Implementation Insights

This repository is aiming to show how to implement neural network from scratch
along with diving into the math behind.

**ml-impl-insights** is a repository dedicated to demonstrating how to implement fundamental neural networks from scratch while diving deeply into the underlying mathematics. Each module in the repository tackles a specific concept so that you can follow along and learn how these models really work under the hood.

---

## Table of Contents
1. [Overview](#overview)  
1. [Repository Structure](#repository-structure)  
1. [Installation](#installation)  
1. [How to Run](#how-to-run)  
1. [License](#license)

---

## Overview
Implementing neural networks from scratch can be incredibly valuable for:
- Gaining a deeper understanding of the math and theory behind them.
- Debugging and optimizing models at a low level.
- Exploring variations of standard architectures to see their effect on performance.

This repository includes:
- **Word2Vec (CBOW w/ Softmax)**: Implementation of the Continuous Bag-of-Words (CBOW) variant.
- **Word2Vec (Skip-Gram w/ Negative Sampling)**: Implementation of the Continuous Skip-Gram variant.
- *(Future modules)* Additional examples and mathematical explorations of other network architectures and NLP concepts.

---

## Installation

This project uses [Poetry](https://python-poetry.org/) to manage dependencies. Make sure you have Poetry installed, then run:

```sh
poetry install
```

---

## How to run

Once you’ve installed the dependencies and activated the virtual environment, you can run the CBOW example with:

```sh
poetry run python src/Word2Vec/CBOW.py
```

---

## License

This project is licensed under the MIT License

---

Happy Coding!
