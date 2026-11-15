
import numpy as np

# Function to calculate scaled dot-product attention
def scaled_dot_product_attention(Q, K, V):

    # Step 1: Calculate attention scores using Q and K transpose
    scores = np.matmul(Q, K.T)

    # Step 2: Scale scores by square root of key dimension
    d_k = K.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)

    # Step 3: Apply softmax to get attention weights
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 4: Multiply attention weights with V
    output = np.matmul(attention_weights, V)

    return attention_weights, output


# Input matrices
Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# Call the function
weights, output = scaled_dot_product_attention(Q, K, V)

# Print the results
print("Attention Weights:")
print(np.round(weights, 3))

print("\nFinal Output:")
print(np.round(output, 3))
