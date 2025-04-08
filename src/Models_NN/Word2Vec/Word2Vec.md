# Word2Vec Implementation

This folder contains an implementation of **Word2Vec**, one of the most popular neural word embedding methods. Word2Vec learns vector representations of words by predicting surrounding words given a target word (or vice versa), capturing semantic and syntactic relationships in a continuous vector space.

## Table of Contents
1. [Introduction](#introduction)
2. [Overview of Variants](#overview-of-variants)
   - [CBOW (Continuous Bag-of-Words)](#cbow-continuous-bag-of-words)
   - [Skip-Gram](#skip-gram)
   - [Training Objectives: Hierarchical Softmax vs. Negative Sampling](#training-objectives-hierarchical-softmax-vs-negative-sampling)
3. [Implementation Details](#implementation-details)
4. [Installation and Setup](#installation-and-setup)
5. [Usage](#usage)
6. [Examples and Tutorials](#examples-and-tutorials)
7. [References](#references)

---

## Introduction
**Word2Vec** is a technique for learning vector representations (embeddings) of words in a way that words sharing similar contexts have vectors close to each other in the embedding space. Developed by Tomas Mikolov and colleagues at Google in 2013, Word2Vec has two core architectures:

- **CBOW (Continuous Bag-of-Words)**: Predicts the target word given its surrounding context.
- **Skip-Gram**: Predicts the surrounding context words given the target word.

Each architecture can be trained in different ways, most commonly using either **hierarchical softmax** or **negative sampling** to approximate the full softmax function.

---

## Overview of Variants
Word2Vec is often summarized as having **four** core variants because of the combination of:
1. **CBOW** or **Skip-Gram**  
2. **Hierarchical Softmax** or **Negative Sampling**

### CBOW (Continuous Bag-of-Words)
- **Key Idea**: Use surrounding words (context) to predict the target (center) word.  
- **Example**: If the context is \[ “The”, “cat”, “on”, “the” \_\_\_ \], the model learns to predict “mat” (if that’s the missing word).  
- **Pros**: Often faster than Skip-Gram for smaller datasets and smooths over context by averaging embeddings of surrounding words.

### Skip-Gram
- **Key Idea**: Use the target (center) word to predict its surrounding context words.  
- **Example**: If the center word is “cat,” predict context words like “The”, “on”, “mat”, etc.  
- **Pros**: Tends to do better on smaller amounts of training data and can capture more nuanced contexts.

### Training Objectives: Hierarchical Softmax vs. Negative Sampling
- **Hierarchical Softmax**: Builds a binary tree of all words in the vocabulary and computes the loss along a path from the root to the target word. This reduces the computational complexity from being proportional to the vocabulary size to the tree depth.  
- **Negative Sampling**: Instead of computing a full softmax, sample a few “negative” words from the vocabulary to update at each step. This is often computationally cheaper and works well in practice.

---

## Implementation Details
1. **Tokenization & Preprocessing**  
   - Convert sentences into a list of tokens (words), handle punctuation, and possibly lowercase.  
   - Optionally remove stopwords or apply sub-sampling techniques to reduce frequent words.

2. **Building the Vocabulary**  
   - Create a mapping of each word to a unique ID (word-to-index).  
   - Track word frequencies if using sub-sampling or negative sampling.

3. **Model Structure**  
   - For **CBOW**: 
     - **Input Layer**: One-hot or index representation of context words.  
     - **Projection Layer**: Embeddings for each word ID.  
     - **Output Layer**: Predict the center word embedding, often via a softmax or negative sampling objective.
   - For **Skip-Gram**: 
     - **Input Layer**: One-hot or index representation of the target word.  
     - **Projection Layer**: Embeddings for each word ID.  
     - **Output Layer**: Predict the context word embeddings, again via hierarchical softmax or negative sampling.

4. **Training**  
   - Typically performed in batches or mini-batches.  
   - Loss functions differ slightly for each variant (hierarchical softmax vs. negative sampling).  
   - Use backpropagation to update embedding weights.

5. **Hyperparameters**  
   - **Embedding Size**: Usually in the range of 100 to 300 for many applications, but can vary.  
   - **Window Size**: How many words to the left/right form the context.  
   - **Learning Rate**: Often decreases over time.  
   - **Epochs**: Can vary based on dataset size and complexity.

---

## Installation and Setup
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo/src/Word2Vec
