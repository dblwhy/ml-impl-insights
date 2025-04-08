# CBOW (Continuous Bag-of-Words)

## Overview

CBOW predicts a target word based on its surrounding context words. 

For example, if the context (input word) is \[ \_\_\_, "is", "the", "highest", "mountain", “in", "Japan"], the model learns to predict "Mt.Fuji", not "Everest" or "What".

The name "Continuous Bag of Words" comes from treating the context words as a bag (unordered set) of words.

Since this algorithm itself is pretty simple, it is not realistic to use for sentence generation like ChatGPT but for word prediction. Following examples would be one of the practical usecases.

Ex1. Improving search engines
- Querying documents which includes semantically similar words
- Training example:
  - Context: ["wireless", "noise", "canceling", "headphones"], Target: "earphones"
  - Context: ["how", "to", "setup"], Target: "onboarding"

Ex2. Missing Word Prediction:
- Suggesting next words or correcting the sentence while typing
- Training example:
  - Context: ["as", "soon", "as"], Target: "possible"
  - Context: ["shipping", "to", "united", "of", "america"], Target: "states"

## Implementation

CBOW implementation follows the following steps such as the other neural network. It'll repeat the steps for "every training data * epochs count (=the count to redo the entire process again)" times. Note that people call the step 1-5 as "forward propagation" and last step as "back propagation".
1. Prepare input data
1. Dot product (≒multiply) the input data by weight
1. Dot product the output of step 2 by another weight
1. Use Softmax function to determine the probability
1. Calculate the error
1. Modify the weights (back propagation)

## Forward Propagation Implementation

This is the visualized example of Forward Propagation.

![CBOW Forward Propagation](/document/images/word2vec-forward-propagation.jpg "Forward Propagation")

### Pre-Requisite

In CBOW model, we first need to gather every single words in the database and assign an ID to it.
Since we need to convert each word into an integer to do the calculation, we use one-hot-encoding.
one-hot-encoding is basically an array which can only contain 0 or 1.

Let's say there are 10000 words in total in all the documents you want to feed to AI. 
Then, we create an array which length is 10000 and define a rule which index represents which word.
For example, your rule could be index=0 is mapped to "apple" and index=9999 is mapped to "x-ray" (if those words exists in your document).
And when you need to create an one-hot-encoding array for "apple", it looks like only index=0 is set to 1 and the rest is 0.

And what is the final output looks like?
Final output will also be a 10000 length array, but stores probability inside. You can see it says the answer is "apple"(index:0) for `1%` probability and "x-ray"(index:9999) is `21%` probability. Then, you pick the word which has the highest probability.
```python
[0.001, 0.5, 0.02, ..., 0.90, 0.07, 0.21]
```

### Step 1

Let's think about a case which Context is ["is", "the", "highest", "mountain", “in", "Japan"] and Target is "Mt.Fuji".
First we need to prepare 6 one-hot-encoding arrays for each words, the implementation will looks like below.

```python
# @context_words: 1-D array for context words
# @vocab_size: total words exists in the database
def one_hot_vectors(context_words, vocab_size):
    vectors = np.zeros((len(context_words), vocab_size))
    for i, word in enumerate(context_words):
        vectors[i, word_to_index[word]] = 1
    return vectors
```

### Step 2

We simply sum the input array and take the mean. This operation allow us to increase the weight for the repetitive words.
We can call this mean array as an "Input Layer" in terms of Neural Network.

```python
context_vectors = one_hot_vectors(context_words, VOCAB_SIZE)
mean_context_vector = np.mean(context_vectors, axis=0, keepdims=True) 
```

### Step 3

Calculate the dot product between the "Input Layer" and "Weight(=W1)". The output array typically named as "Hidden Layer".

```python
W1 = np.random.randn(VOCAB_SIZE, EMBEDDING_DIM) * 0.01 

hidden_layer = np.dot(mean_context_vector, W1)
```

Initially, `W1` is just an globally defined 2-d array containing full of random numbers which range is around -0.03-0.03. And it will be modified during its Back Propagation process.

Also the size of `W1` is (N, 50) where N means the total vocab size and 50 is a random number that I've just chosen (in my understanding, people often choose around 50-300 for CBOW as a best practice).

This dot product calculation is aiming to reduce the input size.

It could be easier to think of the matrix rule that when you dot product between (l, m) sized matrix and (m, n) size matrix,
the size of the result matrix will become (l, n). The common part `m` has removed and output size becomes row:`l` and col:`n`. 

In real world, English has more than 1 million words, the size of the array shown in the Input Layer will become huge.
So we can reduce its size into (1, 50) by calculating (1, N)・(N, 50).
Also it is worth noting that since most of the item is 0 in the Input Layer array, most of the weight multiplication becomes 0 as well. Thus the outcome matrix only includes meaningful value, it has transformed into dense and low-dimensional space.

### Step 4

Calculate the dot product between the "Hidden Layer" and "Weight(=W2)".

```python
output_scores = np.dot(hidden_layer, W2)
```

This calculation is aiming to increase the dimension back to (1, N) from the high-dense matrix we prepared in the previous step. This step is required because our final output is an array which length is N.

### Step 5

Convert the array full of random numbers into probabilities. There is a wonderful function which is called Softmax function which can convert this kind data into probability.

```python
exp_scores = np.exp(output_scores)
probs = exp_scores / np.sum(exp_scores)
```

The math equation of Softmax looks like below.

$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$

where:

- $z$ is the input vector of real numbers.
- $e^{z_i}$ represents the exponential function applied to each numbers.

It may looks difficult but it is just calculating the ratio.
such as ratio of [1, 2, 3] is [1/6, 2/6, 3/6]. The reason it uses factorial is because output never gets negative such as $e^{-10} = 0.00...$ and $e^{10} = 22026.4...$ . Also the reason it uses exponential is because it is `Identity Function`. Meaning the differential output is same as the input: $e^{\prime}({x}) = e^{x}$. During the Back Propagation process we need to calculate differential many times, which using exponential makes our life easier.

### Step 6

Compute loss using Cross-Entropy equation. People call this equation which intends to calculate the loss as "Loss Function" or "Cost Function".

If you don't like math you can just think `Cross-Entropy equation` will calculate the error between the predicted (1,N) probabilities and one-hot-encoded target word.

```python
target_vector = one_hot_vectors([target_word], VOCAB_SIZE)
loss = -np.sum(target_vector * np.log(probs + 1e-9))  # 1e-9 is just for avoiding log(0) error
```

The Cross-Entropy equation looks like below.

$$
Loss = \sum_{i=1}^{n} p(x) \log(\frac{1}{q(x)})
$$

To understand this equation, you need to understand Shannon Entropy. Shannon Entropy is an equation looks like below which takes logarithm of 1/p instead of 1/q.

$$
Entropy = \sum_{i=1}^{n} p(x) \log(\frac{1}{p(x)})
$$

At that time, he was finding an equation that can result in a large number when something unexpected happens and lower number when something expected happens such as the below graph shows. As you can see, this is taking the logarithm of probability.

![Shannon Entropy Graph](/document/images/word2vec-shannon-entropy.jpg "Shannon Entropy")

Then how do we create the equation of probability? We use 
probability density function. Following is the probability density function of a Gaussian (Normal) distribution which $μ$ is the average number in the distribution, $σ$ is the variance and $x$ is the actual value.

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
$$

In short, this equation says that once we define the shape of the probability distribution using $μ$ and $σ$, it'll tells you how likely the sample data you have in your hand happens. For example, let's think of the relation between the hours spent on studying and the final score he/she gets. Now you have a sample data for student A who studied 10 hours and scored 10/10, we want to know how likely this event could happen. When we already know the average score who studied for 10 hours is 6/10, the above equation will shows us a lower probability compared to whom taken 6/10.

Therefore, once we use this probability density function in Shannon Entropy, it'll tells us the Expected Value of the surprise we get from the sample data we have.

Now let's get back to the topic. Cross-Entropy is using both $p(x)$ and $q(x)$ and $p(x)$ is the probability density function for the target value and $q(x)$ is the function to determine the prediction (Softmax function in this example). And it is calculating the Expected Value using both probabilities. It could be easier to think, it is showing the Entropy when we are believing in our prediction but the reality was different. So our goal is to see the error is close enough to zero when we subtract pure Entropy of p(x) from the Cross Entropy. In the word we need to make Cross Entropy value has settled to its minimum value during the machine learning process.

### Step 7

Modify W2 numbers based on the error we get at previous step.

Again, if you don't like math you can just remember if we use Softmax at the Output Layer and Cross-Entropy as Loss function, 
you can simply use the diff between predicted probabilities and target one-hot encoded array.

```python
dE_dz = probs - target_vector  
dL_dW2 = np.dot(hidden_layer.T, dE_dz)

W2 -= LEARNING_RATE * dL_dW2
```

How do we modify the weight?

In the weight matrix, it includes many weights and we need to modify each weights carefully. The idea is to understand how the specific weight has contributed to the final error.

So we basically differentiate the error with $ω_{lij}$ which $i$ and $j$ is the specific coordinate in the matrix and $l$ represents which layer the weight is applied to. Since differentiation is same as calculating the gradient of the tangent line, we'll understand whether we should increase the weight or not.

![Gradient of tangent line](/document/images/word2vec-gradient-descent.jpg "Gradient of tangent line")

For example, differentiate the error with $ω_{lij}$ means something like above image. We are not sure how this entire graph looks like but at least we know current weight's gradient $a$ is a positive value. Thus, it indicates that if we decrease the weight, the error will decrease. On the other hand, if the differentiated value is negative, we need to increase the weight to decrease the error. By repeating this process until the gradient becomes zero, we can optimize the weight. We call this approach "Gradient Descent" and it is widely used across various engineering fields.

However, as you can imagine, we are not sure that the point we have settled is the minimum value or not. Maybe the graph is going up and down and there could be a better weight which minimizes the error. This is a challenge in Gradient Descent and there are many variants to mitigate this issue.

Now let's try to calculate the differentiation between the error and weight in our example.

$$
\frac{\partial E}{\partial w^{(L)}_{i,j}}
= 
\frac{\partial z^{(L)}_i}{\partial w^{(L)}_{i,j}}
\,\frac{\partial a^{(L)}_i}{\partial z^{(L)}_i}
\,\frac{\partial E}{\partial a^{(L)}_i}
$$

When:

$$
z^{(L)}_i = w^{(L)}_{i,j}a^{(L-1)}_i
$$
$$
a^{(L)}_i = softmax(z^{(L)}_i) = \frac{e^{z_i}}{\sum_{k=1}^{n} e^{z_k}}
$$
$$
Loss = \sum_{i=1}^{n} t_i \log(\frac{1}{a^{(L)}_i})
$$

Which:\
$E$: Error\
$w^{(L)}_{i,j}$: Specific weight (row:$i$, col:$j$ in weight matrix) used to calculate Layer $L$ (=Output Layer)\
$z_i$: Result of dot product between $w_{i,j}$ and $a^{(L-1)}_i$ (output of the hidden layer at index:$i$ in)\
$a^{(L)}_i$: Activation function (Softmax in this case) used at index:$i$ in the Output Layer\
$t_i$: Target value at index:$i$ (One-hot Encoded target word in our case)

As you can see there is no $w$ inside the Cross Entropy loss function but it is nested. In this case the relation between the Error and specific weight in W2 can be represented using "chain rule". We basically multiply everything what has happened step by step. Now we need to calculate each result:

$$
\frac{\partial E}{\partial a^{(L)}_i}
= \frac{\partial}{\partial a^{(L)}_i}
\left(
    t_1log(\frac{1}{a^{(L)}_1})+...+
    t_nlog(\frac{1}{a^{(L)}_n})
\right)
= -\frac{t_i}{a^{(L)}_i}
$$

$$
\,\frac{\partial a^{(L)}_i}{\partial z^{(L)}_i}
= softmax^{\prime}(z^{(L)}_i) =
\begin{cases}
  a_i (1 - a_i) & (i = k),\\
  -a_ia_k & (i \neq k),\\
\end{cases}
$$

$$
\frac{\partial z^{(L)}_i}{\partial w^{(L)}_{i,j}}
= a^{(L-1)}_i
$$

Then,

$$

\begin{align}
\frac{\partial E}{\partial z^{(L)}_i}
&= -\frac{t_i}{a_i} \cdot a_i(1−a_i) + \left(-\frac{t_i}{a_i}\right) \cdot \sum_{i \neq k} -a_ia_k \notag \\
&= a_i - t_i \notag \\
\end{align}
$$

Therefore,

$$
\therefore\frac{\partial E}{\partial w^{(L)}_{i,j}}
= a^{(L-1)}_i \cdot (a^{(L)}_i - t_i)
$$

Finally, we got the gradient of tangent line.  Last but not least, we need to flip the sign of the gradient and subtract/add to the original weight to decrease the weight. Also we typically multiply the negated gradient with a constant weight to scale the value to a reasonable number.

```python
W2 -= LEARNING_RATE * dL_dW2
```

### Step 8

Modify W1 numbers such as we have done in the previous step.

```python
dL_dh = np.dot(dE_dz, W2.T)
dL_dW1 = # ?

W1 -= LEARNING_RATE * dL_dW1
```

Most of the calculation is quite similar but its a bit different for the hidden layers.

