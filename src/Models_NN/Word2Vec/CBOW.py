import numpy as np

# Define hyperparameters
VOCAB_SIZE = 1000  # Assume 10,000 words in this vocabulary. In terms of English, there are 1 million words in reality
EMBEDDING_DIM = 50  # Embedding size, usually 50-300 for CBOW
CONTEXT_SIZE = 2    # Two words before and after the target
LEARNING_RATE = 0.01
EPOCHS = 2000

# Initialize weight matrices (random small values)
W1 = np.random.randn(VOCAB_SIZE, EMBEDDING_DIM) * 0.01  # (10000, 50) - Input to hidden
W2 = np.random.randn(EMBEDDING_DIM, VOCAB_SIZE) * 0.01  # (50, 10000) - Hidden to output

print(W1)

# Sample training data (word index mapping)
word_to_index = {'I': 0, 'love': 1, 'machine': 2, 'learning': 3, 'and': 4, 'deep': 5, 'neural': 6, 'networks': 7}
index_to_word = {i: word for word, i in word_to_index.items()}
sample_data = [
    (["I", "love", "learning", "and"], "machine"),
    (["love", "machine", "and", "deep"], "learning"),
    (["machine", "learning", "deep", "neural"], "and"),
]

# Function to create one-hot vectors for all context words
# If there are only 5 words in English and when the passed training data is 
# "I love cat", we first assign an ID to each word such as 0:I, 1: love 2: cat.
# Then we make an one hot encoding array for each words such as:
# [
#   [1, 0, 0, 0, 0],   # I
#   [0, 1, 0, 0, 0],   # love
#   [0, 0, 1, 0, 0],   # cat
# ]
def one_hot_vectors(context_words, vocab_size):
    vectors = np.zeros((len(context_words), vocab_size))
    for i, word in enumerate(context_words):
        vectors[i, word_to_index[word]] = 1
    return vectors

# Because each one-hot-encoding arrays are too large and most of them would be zero,
# we want to reduce the size using Matrix. When A is (n, m) and B is (m, l), the size of C when applying A・B = C  will become (n, l).
# By using this symptom, when the one-hot-encoding length is 100000 when can decrease it's size like (1, 100000)・(100000, 50) = (1, 50).
# This "50" is a random number we have to tune, but in terms of CBOW, we use 50-300 size matrix in general.
# We prepare random numbers as a weight to the one-hot-encoding,
# so the result will be transformed value of one-hot-encoding array. Additionally, zero value in one-hot-encoding array, will become zero, so the
# output is a calculated value only using a meaningful value.
# At last we need to calculate the average because each items in the matrix has added up N times when N is the occurrence of the word which could
# get larger when occurrence increases.
# This is in general generating the 2nd layer in neural network 
def hidden_layer(context_vector):
    return np.dot(context_vector, W1) / CONTEXT_SIZE

# Training CBOW model
for epoch in range(EPOCHS):
    total_loss = 0
    for context_words, target_word in sample_data:
        # Convert context words to a single one-hot vector
        context_vectors = one_hot_vectors(context_words, VOCAB_SIZE)  # Shape (1, 10000)
        print(context_vectors.size)

        # Average the context vectors into a single input vector
        mean_context_vector = np.mean(context_vectors, axis=0, keepdims=True)  # Shape (1, vocab_size)

        # Forward propagation
        # Get embeddings for each context word
        hidden_layer = np.dot(mean_context_vector, W1)  # Shape (1, embedding_dim)
        print(hidden_layer)
        
        output_scores = np.dot(hidden_layer, W2)  # (1, 10000)
        exp_scores = np.exp(output_scores)
        probs = exp_scores / np.sum(exp_scores)  # Softmax probabilities

        # Compute loss (-log likelihood)
        target_vector = one_hot_vectors([target_word], VOCAB_SIZE)
        loss = -np.sum(target_vector * np.log(probs + 1e-9))  # 1e-9 is just for avoiding log(0) error
        total_loss += loss

        # Backpropagation
        dE_dz = probs - target_vector  # Gradient w.r.t. output. differential of softmax with cross-entropy simplifies to prob-y
        dL_dW2 = np.dot(hidden_layer.T, dE_dz)  # (50, 1)・(1, 10000) = (50, 10000)

        dL_dh = np.dot(dE_dz, W2.T)  # Gradient w.r.t. hidden layer
        # Gradient w.r.t. W1 (distribute gradients to each context word)
        dL_dW1 = np.zeros_like(W1)
        # for i, word in enumerate(context_words):
        #     word_idx = word_to_index[word]
        #     # Scale gradient by number of context words
        #     dL_dW1[word_idx] += dL_dh[0] / len(context_words)

        # Update weights using gradient descent
        W2 -= LEARNING_RATE * dL_dW2
        W1 -= LEARNING_RATE * dL_dW1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

def predict(context_words):
    context_vectors = one_hot_vectors(context_words, VOCAB_SIZE)
    embeddings = np.dot(context_vectors, W1)
    hidden_layer = np.mean(embeddings, axis=0, keepdims=True)
    output_scores = np.dot(hidden_layer, W2)
    predicted_index = np.argmax(output_scores)
    return index_to_word[predicted_index]

# Test prediction
test_context = ["love", "machine", "and", "deep"]
print(f"Predicted word: {predict(test_context)}")
