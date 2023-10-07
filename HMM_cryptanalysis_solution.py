
import string
import numpy as np
import random
from collections import Counter
from itertools import permutations
import random



# Step 0: Encryption
with open('/Users/shivanshchhabra/Desktop/words (4).txt', 'r') as file:
    plaintext = file.read()

# Generate a random key
alphabet = list(string.ascii_lowercase)
random.shuffle(alphabet)
key = ''.join(alphabet)

# Create a dictionary for the substitution cipher
cipher_dict = {}
for i, char in enumerate(key):
    cipher_dict[chr(i + ord('a'))] = char

# Apply the substitution cipher to the plaintext
ciphertext = ''
for char in plaintext:
    if char.isalpha():
        ciphertext += cipher_dict[char.lower()]
# Step 1: Generate a digraph frequency matrix A for English text
df='/Users/shivanshchhabra/Desktop/english_text_clean.txt' # 1,000,000
with open(df, "r") as f:
    text = f.read().replace('\n', '')

# Initialize the digraph frequency matrix A
A = np.zeros((26, 26))
for i in range(len(text) - 1):
    if text[i].isalpha() and text[i + 1].isalpha():
        row = ord(text[i].lower()) - ord('a')
        col = ord(text[i + 1].lower()) - ord('a')
        A[row, col] += 1

# Add 5 to each element in A
A += 5

# Normalize A
A = A / np.sum(A, axis=1)[:, np.newaxis]


# Step 2: Train an HMM with M = N = 26
def initialize_B_pi():
    B = np.random.rand(26, 26)
    B = B / np.sum(B, axis=1)[:, np.newaxis]
    pi = np.random.rand(26)
    pi = pi / np.sum(pi)
    return B, pi


def train_HMM(O, A, B, pi, num_iters):
    N = A.shape[0]
    T = len(O)
    alpha = np.zeros((T, N))
    beta = np.zeros((T, N))
    gamma = np.zeros((T, N))
    xi = np.zeros((T - 1, N, N))

    for iter in range(num_iters):
        # E-step
        alpha[0, :] = pi * B[:, O[0]]
        alpha[0, :] = alpha[0, :] / np.sum(alpha[0, :])
        for t in range(1, T):
            alpha[t, :] = alpha[t - 1, :] @ A * B[:, O[t]]
            alpha[t, :] = alpha[t, :] / np.sum(alpha[t, :])
        beta[T - 1, :] = 1
        for t in range(T - 2, -1, -1):
            beta[t, :] = A @ (B[:, O[t + 1]] * beta[t + 1, :])
            beta[t, :] = beta[t, :] / np.sum(beta[t, :])
        gamma = alpha * beta
        for t in range(T - 1):
            xi[t, :, :] = alpha[t, :] * A * B[:, O[t + 1]] * beta[t + 1, :]
            xi[t, :, :] = xi[t, :, :] / np.sum(xi[t, :, :])

        # M-step

        pi = gamma[0, :] / np.sum(gamma[0, :])
        A = np.sum(xi, axis=0) / np.sum(gamma[:-1, :], axis=0)[:, np.newaxis]
        for j in range(N):
            B[j, :] = np.sum(gamma[:, j][:, np.newaxis] * (O == j), axis=0) / np.sum(gamma[:, j])

    return A, B, pi

# Step 3: Analyze Result
def compute_putative_key(B, cipher_dict):
    putative_key = {}
    for i, row in enumerate(B):
        putative_key[chr(i + ord('a'))] = cipher_dict[chr(np.argmax(row) + ord('a'))]
    return putative_key

def calculate_accuracy(putative_key, cipher_dict):
    correct_count = 0
    for key, value in putative_key.items():
        if value == cipher_dict[key]:
            correct_count += 1
    return correct_count / len(putative_key)
# Prepare the observation sequence
O = [ord(char.lower()) - ord('a') for char in ciphertext[:1000]]

# Initialize B and pi
B, pi = initialize_B_pi()

# Train the HMM
A, B, pi = train_HMM(O, A, B, pi, num_iters=200)

# Compute the putative key
putative_key = compute_putative_key(B, cipher_dict)

# Calculate the accuracy
accuracy = calculate_accuracy(putative_key, cipher_dict)

print(f"Accuracy: {accuracy:.4f}")

# If accuracy is low, you can try multiple runs with different initializations for B and pi
if accuracy < 0.8:
    num_tries = 10
    best_accuracy = accuracy
    best_putative_key = putative_key

    for _ in range(num_tries):
        # Initialize B and pi
        B, pi = initialize_B_pi()

        # Train the HMM
        A, B, pi = train_HMM(O, A, B, pi, num_iters=200)

        # Compute the putative key
        putative_key = compute_putative_key(B, cipher_dict)

        # Calculate the accuracy
        accuracy = calculate_accuracy(putative_key, cipher_dict)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_putative_key = putative_key

    print(f"Best Accuracy: {best_accuracy:.4f}")
# Calculate the confusion matrix
true_key = [cipher_dict[chr(i + ord('a'))] for i in range(26)]
predicted_key = [putative_key[chr(i + ord('a'))] for i in range(26)]
cm = confusion_matrix(true_key, predicted_key)

# Plot the confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=list(string.ascii_lowercase), yticklabels=list(string.ascii_lowercase))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()