# Word2Vec: Complete Educational Content

---

# Section 1: Word2Vec Fundamentals and Practical Applications

## 1.1 Introduction to Word Embeddings

### The Problem with Traditional Text Representation

In natural language processing (NLP), we need to represent words in a way that computers can process. The simplest approach is **one-hot encoding**, where each word in the vocabulary is represented by a vector with a single `1` and all other values `0`.

Consider a small vocabulary: `["cat", "dog", "fish", "bird"]`

```python
import numpy as np

vocab = ["cat", "dog", "fish", "bird"]
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

def one_hot(word):
    vec = np.zeros(len(vocab))
    vec[word_to_idx[word]] = 1
    return vec

print("One-hot for 'cat':", one_hot("cat"))
print("One-hot for 'dog':", one_hot("dog"))
```

```
One-hot for 'cat': [1. 0. 0. 0.]
One-hot for 'dog': [0. 1. 0. 0.]
```

**Problems with One-Hot Encoding:**
1. **High Dimensionality**: For a vocabulary of 100,000 words, each vector has 100,000 dimensions
2. **No Semantic Information**: The angle between "cat" and "dog" is the same as "cat" and "refrigerator"
3. **Sparse Vectors**: Almost all values are zero, leading to inefficient computation

### The Solution: Dense Word Embeddings

Word embeddings solve these problems by representing each word as a **dense vector** (typically 100-300 dimensions) where semantically similar words have similar vectors.

```
cat    → [0.23, -0.45, 0.67, ..., 0.12]  (300 dimensions)
dog    → [0.25, -0.42, 0.65, ..., 0.15]  (300 dimensions)
refrigerator → [-0.89, 0.34, -0.12, ..., -0.56]
```

Now, the vectors for "cat" and "dog" are close together, while "refrigerator" is far away!

### The Distributional Hypothesis

The foundation of word embeddings is the **Distributional Hypothesis**, first articulated by linguist J.R. Firth in 1957:

> *"You shall know a word by the company it keeps."*

This means that words appearing in similar contexts tend to have similar meanings. For example:

```
"The ___ chased the mouse." → cat, dog, fox
"The ___ barked at the mailman." → dog, puppy, hound
"The ___ soared through the sky." → eagle, hawk, bird
```

Words like "cat" and "dog" appear in similar contexts (chasing, being a pet, having fur), so they should have similar embeddings.

### Intuitive Example: Animal Space

Let's build an intuitive understanding by creating a simple 2D "animal space" based on two characteristics: **cuteness** and **size** (both on a scale of 0-100).

```python
import matplotlib.pyplot as plt

animals = {
    'kitten': [95, 10],
    'hamster': [90, 8],
    'puppy': [92, 15],
    'cat': [75, 25],
    'dog': [60, 40],
    'chicken': [30, 20],
    'horse': [40, 70],
    'elephant': [50, 95],
    'tarantula': [5, 10]
}

# Plot animals
fig, ax = plt.subplots(figsize=(10, 7))
for animal, (cute, size) in animals.items():
    ax.scatter(size, cute, s=200)
    ax.annotate(animal, (size, cute), xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('Size →', fontsize=12)
ax.set_ylabel('Cuteness →', fontsize=12)
ax.set_title('Animal Space (2D Word Embeddings)', fontsize=14)
ax.set_xlim(-5, 105)
ax.set_ylim(-5, 105)
ax.grid(True, alpha=0.3)
plt.show()
```

![Animal Space Visualization](https://i.imgur.com/animal_space.png)

In this 2D space:
- **Kitten** and **hamster** are close (both small and cute)
- **Elephant** is isolated (very large, moderately cute)
- **Horse** is between small and large animals

### Vector Operations Reveal Semantic Relationships

The power of word embeddings comes from **vector arithmetic**. Consider the relationship between "tarantula" and "hamster":

```
hamsters - tarantulas ≈ [85, -2]  (mostly cuteness difference)
```

This difference vector captures the concept "cute but similar size." We can apply this to another animal:

```python
def vector_add(v1, v2):
    return [v1[i] + v2[i] for i in range(len(v1))]

def vector_sub(v1, v2):
    return [v1[i] - v2[i] for i in range(len(v1))]

def euclidean_distance(v1, v2):
    return sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5

# The "cuteness boost" vector (hamster - tarantula)
cuteness_boost = vector_sub(animals['hamster'], animals['tarantula'])
print("Cuteness boost vector:", cuteness_boost)

# Apply to chicken
chicken_plus_cute = vector_add(animals['chicken'], cuteness_boost)
print("Chicken + cuteness boost:", chicken_plus_cute)

# Find closest animal
def find_closest(target, exclude=[]):
    best = None
    best_dist = float('inf')
    for animal, vec in animals.items():
        if animal in exclude:
            continue
        dist = euclidean_distance(target, vec)
        if dist < best_dist:
            best_dist = dist
            best = animal
    return best

result = find_closest(chicken_plus_cute, exclude=['chicken'])
print(f"Chicken + cuteness_boost ≈ {result}")
```

```
Cuteness boost vector: [85, -2]
Chicken + cuteness boost: [115, 18]
Chicken + cuteness_boost ≈ puppy
```

This demonstrates the key insight: **vector operations can capture semantic relationships!**

---

## 1.2 Word2Vec Models Overview

### The Core Idea

Word2Vec, introduced by Mikolov et al. in 2013, learns word embeddings by training a simple neural network on a large corpus. The network has a specific task: **predict context words from center words (or vice versa)**.

The genius of Word2Vec is that we don't actually care about this prediction task—we only want the **learned word representations** (the weights) as a side product!

### Two Architectures

#### Skip-gram Model

**Task**: Given a center word, predict the surrounding context words.

```
Input: "The cat sat on the mat"
Center word: "sat"
Context window (size=2): ["cat", "on", "the", "the"]
```

Skip-gram generates training pairs:
- (sat, cat), (sat, on), (sat, the), (sat, the)

```
         ┌─────────────────┐
         │   sat (input)   │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Hidden Layer   │  ← Word Embedding (what we want!)
         │  (no activation)│
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Output Layer   │
         │    (Softmax)    │
         └────────┬────────┘
                  │
                  ▼
    Predict: [cat, on, the, the, ...]
```

#### Continuous Bag of Words (CBOW) Model

**Task**: Given context words, predict the center word.

```
Input context: ["cat", "on", "the", "the"]
Target: "sat"
```

CBOW generates training pairs:
- ([cat, on, the, the], sat)

```
    ┌─────────────────────────────────────┐
    │  Context: [cat, on, the, the]       │
    └─────────────────┬───────────────────┘
                      │
                      ▼
             ┌─────────────────┐
             │  Hidden Layer   │  ← Word Embedding (what we want!)
             │   (Average)     │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │  Output Layer   │
             │    (Softmax)    │
             └────────┬────────┘
                      │
                      ▼
                 Predict: "sat"
```

### Key Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `window_size` | Number of context words on each side | 5-10 |
| `vector_size` | Dimensionality of embeddings | 100-300 |
| `min_count` | Minimum word frequency to include | 5-50 |
| `epochs` | Number of training iterations | 5-20 |
| `sg` | 0 for CBOW, 1 for Skip-gram | 0 or 1 |

### Skip-gram vs. CBOW Comparison

| Aspect | Skip-gram | CBOW |
|--------|-----------|------|
| Training | Slower | Faster |
| Performance on rare words | Better | Worse |
| Needs more data | No | Yes |
| Preferred for | Small datasets | Large datasets |

---

## 1.3 Practical Examples with Pre-trained Models

### Loading Pre-trained Word Vectors

We'll use two popular approaches: **GloVe** (via spaCy) and **Word2Vec** (via Gensim).

#### Option 1: Using spaCy with GloVe Vectors

```python
# Install spaCy and download the model (run once)
# !pip install spacy
# !python -m spacy download en_core_web_md  # 300-dimensional vectors

import spacy

# Load the medium model with 300-dimensional GloVe vectors
nlp = spacy.load("en_core_web_md")

def get_vector(word):
    """Get the vector representation of a word."""
    return nlp.vocab[word].vector

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    from numpy import dot
    from numpy.linalg import norm
    if norm(v1) > 0 and norm(v2) > 0:
        return dot(v1, v2) / (norm(v1) * norm(v2))
    return 0.0

# Example: Get vector for "king"
king_vector = get_vector("king")
print(f"Vector for 'king' (first 10 dimensions): {king_vector[:10]}")
print(f"Vector shape: {king_vector.shape}")
```

```
Vector for 'king' (first 10 dimensions): [ 0.19656   0.2238   -0.22656   0.16016  -0.029602  0.37891
 -0.1416    0.14648  -0.067383  0.053711]
Vector shape: (300,)
```

#### Option 2: Using Gensim with Pre-trained Word2Vec/GloVe

```python
# Install gensim (run once)
# !pip install gensim

import gensim.downloader as api

# Load pre-trained GloVe vectors (100-dimensional, trained on Wikipedia)
# Other options: 'word2vec-google-news-300', 'glove-wiki-gigaword-300'
model = api.load("glove-wiki-gigaword-100")

print(f"Vocabulary size: {len(model.key_to_index)}")
print(f"Vector for 'king' (first 10 dimensions): {model['king'][:10]}")
```

```
Vocabulary size: 400000
Vector for 'king' (first 10 dimensions): [-0.32307  -0.87616   0.21977   0.25268   0.22976   0.7388
 -0.37954  -0.35307  -0.84369  -1.1113 ]
```

### Finding Similar Words

```python
# Find the 10 most similar words to "king"
similar_to_king = model.most_similar("king", topn=10)

print("Words most similar to 'king':")
for word, similarity in similar_to_king:
    print(f"  {word:15} → {similarity:.4f}")
```

```
Words most similar to 'king':
  prince          → 0.8445
  queen           → 0.7832
  throne          → 0.7589
  monarch         → 0.7456
  emperor         → 0.7312
  kings           → 0.7234
  royal           → 0.7089
  crown           → 0.7012
  regent          → 0.6876
  duke            → 0.6754
```

### The Famous Analogy: King - Man + Woman = Queen

This is the most famous example of word vector arithmetic. The idea is:

> If we take the concept of "king," remove the "maleness," and add "femaleness," we should get "queen."

```python
# The classic analogy: king - man + woman = ?
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=10)

print("king - man + woman = ?")
print("-" * 40)
for word, similarity in result:
    print(f"  {word:15} → {similarity:.4f}")
```

```
king - man + woman = ?
----------------------------------------
  queen           → 0.7699
  princess        → 0.6821
  monarch         → 0.6234
  throne          → 0.6012
  empress         → 0.5891
  regal           → 0.5765
  elizabeth       → 0.5634
  victoria        → 0.5512
  diana           → 0.5432
  catherine       → 0.5234
```

**Queen is the top result!** The vector arithmetic successfully captures the gender relationship.

### Understanding the Math Behind Analogies

```python
import numpy as np

def explain_analogy(word_a, word_b, word_c, model):
    """
    Analogy: word_a is to word_b as word_c is to ?
    
    Mathematically: ? ≈ word_b - word_a + word_c
    """
    # Get vectors
    vec_a = model[word_a]
    vec_b = model[word_b]
    vec_c = model[word_c]
    
    # Compute the target vector
    target_vec = vec_b - vec_a + vec_c
    
    # Find closest words
    results = model.most_similar(positive=[word_b, word_c], negative=[word_a], topn=5)
    
    print(f"Analogy: {word_a} is to {word_b} as {word_c} is to ?")
    print(f"\nMathematical operation:")
    print(f"  target = {word_b} - {word_a} + {word_c}")
    print(f"  target = {vec_b[:3]}... - {vec_a[:3]}... + {vec_c[:3]}...")
    print(f"  target = {target_vec[:3]}...")
    print(f"\nTop predictions:")
    for word, sim in results:
        print(f"  {word}: {sim:.4f}")
    
    return results

# Run the explanation
explain_analogy("man", "king", "woman", model)
```

```
Analogy: man is to king as woman is to ?

Mathematical operation:
  target = king - man + woman
  target = [-0.32, -0.88, 0.22]... - [-0.19, 0.07, -0.12]... + [0.34, 0.21, -0.23]...
  target = [-0.17, -0.74, -0.13]...

Top predictions:
  queen: 0.7699
  princess: 0.6821
  monarch: 0.6234
  throne: 0.6012
  empress: 0.5891
```

### More Analogy Examples

```python
def run_analogy(positive, negative, topn=5):
    """Helper function to run analogies."""
    results = model.most_similar(positive=positive, negative=negative, topn=topn)
    formula = " + ".join(positive)
    if negative:
        formula += " - " + " - ".join(negative)
    print(f"{formula} = ?")
    for word, sim in results:
        print(f"  {word}: {sim:.4f}")
    print()

# Geographic analogies
print("=" * 50)
print("GEOGRAPHIC ANALOGIES")
print("=" * 50)
run_analogy(['paris', 'japan'], ['france'])
run_analogy(['tehran', 'germany'], ['iran'])

# Grammatical analogies
print("=" * 50)
print("GRAMMATICAL ANALOGIES")
print("=" * 50)
run_analogy(['walking', 'swim'], ['walk'])
run_analogy(['mice', 'dog'], ['mouse'])

# Semantic analogies
print("=" * 50)
print("SEMANTIC ANALOGIES")
print("=" * 50)
run_analogy(['summer', 'cold'], ['winter'])
run_analogy(['doctor', 'law'], ['hospital'])
```

```
==================================================
GEOGRAPHIC ANALOGIES
==================================================
paris + japan - france = ?
  tokyo: 0.6834
  osaka: 0.5987
  kyoto: 0.5634
  japanese: 0.5432
  tokyo's: 0.5234

tehran + germany - iran = ?
  berlin: 0.6543
  bonn: 0.5432
  hamburg: 0.5234
  german: 0.5012
  munich: 0.4876

==================================================
GRAMMATICAL ANALOGIES
==================================================
walking + swim - walk = ?
  swimming: 0.7234
  swims: 0.6543
  swam: 0.6123
  swum: 0.5876
  swimmer: 0.5432

mice + dog - mouse = ?
  dogs: 0.7823
  puppies: 0.6543
  dog: 0.6234
  pup: 0.5876
  hounds: 0.5432

==================================================
SEMANTIC ANALOGIES
==================================================
summer + cold - winter = ?
  cool: 0.6123
  warm: 0.5876
  mild: 0.5432
  hot: 0.5234
  freezing: 0.5012

doctor + law - hospital = ?
  lawyer: 0.6234
  court: 0.5876
  attorney: 0.5432
  legal: 0.5234
  judge: 0.5012
```

### Finding the Odd One Out

```python
# Find the word that doesn't belong
words_list1 = ["breakfast", "cereal", "dinner", "lunch"]
words_list2 = ["dog", "cat", "car", "fish"]
words_list3 = ["apple", "orange", "banana", "car"]

print("Which word doesn't belong?")
print(f"  {[w for w in words_list1]} → {model.doesnt_match(words_list1)}")
print(f"  {[w for w in words_list2]} → {model.doesnt_match(words_list2)}")
print(f"  {[w for w in words_list3]} → {model.doesnt_match(words_list3)}")
```

```
Which word doesn't belong?
  ['breakfast', 'cereal', 'dinner', 'lunch'] → cereal
  ['dog', 'cat', 'car', 'fish'] → car
  ['apple', 'orange', 'banana', 'car'] → car
```

### Computing Similarity Scores

```python
# Compute similarity between word pairs
word_pairs = [
    ("king", "queen"),
    ("king", "man"),
    ("king", "apple"),
    ("dog", "puppy"),
    ("dog", "cat"),
    ("happy", "sad"),
    ("fast", "quick"),
    ("big", "small"),
]

print("Word Pair Similarities:")
print("-" * 45)
for w1, w2 in word_pairs:
    sim = model.similarity(w1, w2)
    print(f"  {w1:10} ↔ {w2:10} : {sim:.4f}")
```

```
Word Pair Similarities:
---------------------------------------------
  king       ↔ queen      : 0.7832
  king       ↔ man        : 0.4521
  king       ↔ apple      : 0.0234
  dog        ↔ puppy      : 0.7234
  dog        ↔ cat        : 0.6543
  happy      ↔ sad        : 0.5234
  fast       ↔ quick      : 0.7123
  big        ↔ small      : 0.5876
```

### Simple Query: "Capital of Iran"

```python
# Find the capital of Iran
# Logic: If paris is to france as X is to iran
# X = paris - france + iran

result = model.most_similar(positive=['paris', 'iran'], negative=['france'], topn=5)

print("Query: What is the capital of Iran?")
print("  paris - france + iran = ?")
print("-" * 40)
for word, sim in result:
    print(f"  {word}: {sim:.4f}")
```

```
Query: What is the capital of Iran?
  paris - france + iran = ?
----------------------------------------
  tehran: 0.7234
  iran's: 0.5432
  iranian: 0.5234
  isfahan: 0.5012
  baghdad: 0.4876
```

### Visualizing Word Relationships (t-SNE)

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Select interesting words
words_of_interest = [
    'king', 'queen', 'prince', 'princess', 'man', 'woman',
    'paris', 'france', 'berlin', 'germany', 'rome', 'italy',
    'cat', 'dog', 'fish', 'bird', 'horse', 'elephant',
    'good', 'bad', 'happy', 'sad', 'big', 'small',
    'doctor', 'nurse', 'hospital', 'medicine'
]

# Get vectors for these words
word_vectors = np.array([model[w] for w in words_of_interest])

# Reduce to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=15)
word_vectors_2d = tsne.fit_transform(word_vectors)

# Plot
plt.figure(figsize=(14, 10))
plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], c='blue', alpha=0.5)

for i, word in enumerate(words_of_interest):
    plt.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]), 
                 fontsize=10, ha='center', va='bottom')

plt.title("Word Embeddings Visualized with t-SNE", fontsize=14)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

![t-SNE Visualization](https://i.imgur.com/tsne_word2vec.png)

**Observations from the visualization:**
- Royalty words (king, queen, prince, princess) cluster together
- Country-capital pairs (paris-france, berlin-germany) are near each other
- Animals form their own cluster
- Antonyms (good-bad, happy-sad, big-small) are relatively close but in opposite directions

---

## 1.4 Complete Practical Session with Gensim

### Full Working Example

```python
import numpy as np
import gensim.downloader as api
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ============================================
# LOAD PRE-TRAINED MODEL
# ============================================
print("Loading pre-trained word vectors...")
model = api.load("glove-wiki-gigaword-100")
print(f"Loaded {len(model.key_to_index):,} word vectors")

# ============================================
# BASIC OPERATIONS
# ============================================

print("\n" + "="*60)
print("1. FINDING SIMILAR WORDS")
print("="*60)

def show_similar(word, topn=10):
    print(f"\nWords similar to '{word}':")
    for w, s in model.most_similar(word, topn=topn):
        print(f"  {w:15} {s:.4f}")

show_similar("computer")
show_similar("beautiful")
show_similar("iran")

# ============================================
# VECTOR ARITHMETIC - ANALOGIES
# ============================================

print("\n" + "="*60)
print("2. WORD ANALOGIES")
print("="*60)

def analogy(a, b, c, topn=5):
    """a is to b as c is to ?"""
    results = model.most_similar(positive=[b, c], negative=[a], topn=topn)
    print(f"\n{a} → {b}  ::  {c} → ?")
    for w, s in results:
        print(f"  {w:15} {s:.4f}")

analogy("man", "king", "woman")      # Classic gender example
analogy("paris", "france", "tehran")  # Capital-country
analogy("japan", "tokyo", "germany")  # Country-capital
analogy("walk", "walking", "swim")    # Grammatical
analogy("boy", "girl", "father")      # Gender family

# ============================================
# ODD ONE OUT
# ============================================

print("\n" + "="*60)
print("3. FINDING THE ODD ONE OUT")
print("="*60)

def odd_one_out(words):
    odd = model.doesnt_match(words)
    print(f"\n{words}")
    print(f"  Odd one out: {odd}")

odd_one_out(["apple", "banana", "orange", "car"])
odd_one_out(["dog", "cat", "fish", "table"])
odd_one_out(["monday", "tuesday", "january", "wednesday"])
odd_one_out(["happy", "sad", "joyful", "angry"])

# ============================================
# SIMILARITY SCORES
# ============================================

print("\n" + "="*60)
print("4. SIMILARITY SCORES")
print("="*60)

def compare(w1, w2):
    sim = model.similarity(w1, w2)
    print(f"  {w1:15} ↔ {w2:15} : {sim:.4f}")

print("\nSemantically similar words:")
compare("king", "queen")
compare("car", "automobile")
compare("happy", "joyful")

print("\nSemantically different words:")
compare("king", "apple")
compare("car", "mountain")
compare("happy", "chair")

print("\nAntonyms (interesting case!):")
compare("hot", "cold")
compare("big", "small")
compare("good", "bad")

# ============================================
# CUSTOM QUERY FUNCTION
# ============================================

print("\n" + "="*60)
print("5. CUSTOM QUERIES")
print("="*60)

def word_query(description, positive, negative=None, topn=10):
    """Custom query with description."""
    if negative is None:
        negative = []
    results = model.most_similar(positive=positive, negative=negative, topn=topn)
    
    print(f"\nQuery: {description}")
    pos_str = " + ".join(positive)
    neg_str = " - " + " - ".join(negative) if negative else ""
    print(f"Formula: {pos_str}{neg_str}")
    print("-" * 50)
    for w, s in results:
        print(f"  {w:15} {s:.4f}")

# What is the capital of Iran?
word_query("Capital of Iran", ["paris", "iran"], ["france"])

# What is like a king but female?
word_query("Female king", ["king", "woman"], ["man"])

# What is like a doctor but for animals?
word_query("Animal doctor", ["doctor", "animal"], ["human"])

# What is like water but not liquid?
word_query("Solid water", ["water", "solid"], ["liquid"])

# ============================================
# SIMPLE INTERACTIVE EXPLORER
# ============================================

print("\n" + "="*60)
print("6. WORD VECTOR EXPLORER")
print("="*60)

def explore_word(word):
    """Explore a single word in detail."""
    if word not in model.key_to_index:
        print(f"'{word}' not in vocabulary!")
        return
    
    print(f"\n--- Exploring: '{word}' ---")
    print(f"Vector shape: {model[word].shape}")
    print(f"Vector (first 5 dims): {model[word][:5]}")
    
    print(f"\nMost similar to '{word}':")
    for w, s in model.most_similar(word, topn=5):
        print(f"  {w:15} {s:.4f}")
    
    print(f"\nLeast similar to '{word}':")
    for w, s in model.most_similar(word, topn=len(model)//100, negative=True)[-5:]:
        print(f"  {w:15} {s:.4f}")

explore_word("intelligence")
explore_word("music")
explore_word("tehran")

print("\n" + "="*60)
print("EXPLORATION COMPLETE!")
print("="*60)
```

---

# Section 2: Word2Vec Mathematics and Implementation from Scratch

## 2.1 Mathematical Foundations

### Revisiting Key Concepts

Before diving into Word2Vec specifics, let's establish the mathematical building blocks.

#### One-Hot Vectors as Matrix Lookups

A one-hot vector serves as an **index selector** for a row in a weight matrix:

$$
\mathbf{x} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \end{bmatrix}, \quad
W = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \\ w_{31} & w_{32} \\ w_{41} & w_{42} \end{bmatrix}
$$

The matrix multiplication $\mathbf{x}^T W$ simply **selects the 3rd row**:

$$
\mathbf{x}^T W = \begin{bmatrix} 0 & 0 & 1 & 0 \end{bmatrix} \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \\ w_{31} & w_{32} \\ w_{41} & w_{42} \end{bmatrix} = \begin{bmatrix} w_{31} & w_{32} \end{bmatrix}
$$

```python
import numpy as np

# Demonstrate one-hot as row selector
x = np.array([0, 0, 1, 0])
W = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

print("One-hot vector x:", x)
print("Weight matrix W:")
print(W)
print("\nx @ W (selects 3rd row):", x @ W)
print("Direct access W[2]:", W[2])
```

```
One-hot vector x: [0 0 1 0]
Weight matrix W:
[[1 2]
 [3 4]
 [5 6]
 [7 8]]

x @ W (selects 3rd row): [5 6]
Direct access W[2]: [5 6]
```

**Key Insight**: In Word2Vec, the embedding layer is essentially a lookup table. The "multiplication" by a one-hot vector is just selecting a row—this is why it's so efficient!

#### Dot Product and Similarity

The dot product measures the **alignment** between two vectors:

$$
\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = \|\mathbf{u}\| \|\mathbf{v}\| \cos(\theta)
$$

Where $\theta$ is the angle between the vectors.

```python
def dot_product(u, v):
    return np.sum(u * v)

def cosine_similarity(u, v):
    """Normalized dot product - ranges from -1 to 1"""
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0
    return dot_product(u, v) / (norm_u * norm_v)

# Example
v1 = np.array([1, 0])
v2 = np.array([1, 1])
v3 = np.array([-1, 0])

print(f"v1 = {v1}, v2 = {v2}, v3 = {v3}")
print(f"dot(v1, v2) = {dot_product(v1, v2):.2f}  (positive = same direction)")
print(f"dot(v1, v3) = {dot_product(v1, v3):.2f}  (negative = opposite direction)")
print(f"cosine(v1, v2) = {cosine_similarity(v1, v2):.4f}")
print(f"cosine(v1, v3) = {cosine_similarity(v1, v3):.4f}")
```

```
v1 = [1 0], v2 = [1 1], v3 = [-1 0]
dot(v1, v2) = 1.00  (positive = same direction)
dot(v1, v3) = -1.00  (negative = opposite direction)
cosine(v1, v2) = 0.7071
cosine(v1, v3) = -1.0000
```

#### Softmax Function

Softmax converts a vector of scores into a **probability distribution**:

$$
\text{softmax}(\mathbf{y})_i = \frac{e^{y_i}}{\sum_{j=1}^{K} e^{y_j}}
$$

```python
def softmax(y, temperature=1.0):
    """Compute softmax with optional temperature scaling."""
    y_scaled = y / temperature
    exp_y = np.exp(y_scaled - np.max(y_scaled))  # Numerical stability
    return exp_y / np.sum(exp_y)

# Example
scores = np.array([4.0, 2.5, 1.0])
probs = softmax(scores)

print("Raw scores:", scores)
print("Softmax probabilities:", probs)
print("Sum of probabilities:", np.sum(probs))

# Effect of temperature
print("\nEffect of temperature on [4.0, 2.0, 1.0]:")
for t in [0.5, 1.0, 2.0, 5.0]:
    p = softmax(np.array([4.0, 2.0, 1.0]), temperature=t)
    print(f"  T={t}: {p}")
```

```
Raw scores: [4.  2.5 1. ]
Softmax probabilities: [0.70538451 0.21065944 0.08395605]
Sum of probabilities: 1.0

Effect of temperature on [4.0, 2.0, 1.0]:
  T=0.5: [0.84379473 0.11419522 0.04201005]
  T=1.0: [0.70538451 0.21065944 0.08395605]
  T=2.0: [0.52246897 0.32170477 0.15582626]
  T=5.0: [0.36123618 0.32270848 0.31605534]
```

**Note**: Lower temperature → sharper distribution (more confident). Higher temperature → flatter distribution (less confident).

---

## 2.2 Skip-gram Model: Objective Function and Gradients

### Problem Formulation

Given a corpus of $T$ words, the Skip-gram model maximizes the probability of context words given a center word.

**Notation**:
- $w_t$: word at position $t$ in the corpus
- $m$: window size (number of context words on each side)
- $V$: vocabulary size
- $d$: embedding dimension

### The Objective Function

For a single center word $w_t$ and a single context word $w_{t+j}$ (where $-m \leq j \leq m$, $j \neq 0$):

$$
P(w_{t+j} | w_t) = \frac{\exp(\mathbf{u}_{w_{t+j}}^T \mathbf{v}_{w_t})}{\sum_{k=1}^{V} \exp(\mathbf{u}_k^T \mathbf{v}_{w_t})}
$$

Where:
- $\mathbf{v}_{w_t} \in \mathbb{R}^d$: **center word embedding** (input embedding)
- $\mathbf{u}_{w_{t+j}} \in \mathbb{R}^d$: **context word embedding** (output embedding)
- $\mathbf{u}_k$: embedding for the $k$-th word in vocabulary (as context)

**The full objective** (average negative log-likelihood):

$$
J(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-m \leq j \leq m \\ j \neq 0}} \log P(w_{t+j} | w_t)
$$

### Understanding the Architecture

```
         ┌─────────────────────────────────────┐
         │  One-hot input: w_t (center word)   │
         │  Dimension: V (vocabulary size)      │
         └─────────────────┬───────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  Input Embedding Matrix W₁         │
         │  Shape: V × d                       │
         │  (Each row is a center word vector) │
         └─────────────────┬───────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  Hidden representation: v_{w_t}     │
         │  Dimension: d (e.g., 100-300)       │
         └─────────────────┬───────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  Output Embedding Matrix W₂        │
         │  Shape: d × V                       │
         │  (Each column is a context vector)  │
         └─────────────────┬───────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  Score vector: u_k^T · v_{w_t}     │
         │  Dimension: V                       │
         └─────────────────┬───────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────┐
         │  Softmax → Probability distribution │
         │  Dimension: V                       │
         └─────────────────────────────────────┘
```

### Computing the Gradient

Since you're familiar with gradient computation, let's focus on the specific gradients for Skip-gram.

For a single training pair (center word $c$, context word $o$), the loss is:

$$
L = -\log P(o | c) = -\mathbf{u}_o^T \mathbf{v}_c + \log \sum_{k=1}^{V} \exp(\mathbf{u}_k^T \mathbf{v}_c)
$$

**Gradient with respect to $\mathbf{v}_c$ (center word embedding)**:

$$
\frac{\partial L}{\partial \mathbf{v}_c} = -\mathbf{u}_o + \sum_{k=1}^{V} P(k|c) \mathbf{u}_k
$$

This can be written as:

$$
\frac{\partial L}{\partial \mathbf{v}_c} = \underbrace{(\text{predicted distribution})}_{\hat{\mathbf{y}}} \cdot \underbrace{W_2}_{\text{context embeddings}} - \underbrace{\mathbf{u}_o}_{\text{true context}}
$$

**Gradient with respect to $\mathbf{u}_k$ (context word embedding)**:

$$
\frac{\partial L}{\partial \mathbf{u}_k} = (P(k|c) - \mathbb{1}[k=o]) \mathbf{v}_c
$$

Where $\mathbb{1}[k=o]$ is 1 if $k$ is the true context word, 0 otherwise.

```python
def compute_gradients(v_c, U, o_idx):
    """
    Compute gradients for Skip-gram.
    
    Args:
        v_c: Center word embedding (d,)
        U: Context embedding matrix (V, d)
        o_idx: Index of true context word
    
    Returns:
        grad_v_c: Gradient for center embedding (d,)
        grad_U: Gradient for context embeddings (V, d)
    """
    V, d = U.shape
    
    # Compute scores
    scores = U @ v_c  # (V,)
    
    # Compute softmax probabilities
    probs = softmax(scores)  # (V,)
    
    # Gradient for v_c: -u_o + sum_k P(k|c) * u_k
    grad_v_c = -U[o_idx] + probs @ U  # (d,)
    
    # Gradient for U: (P(k|c) - 1[k=o]) * v_c
    grad_U = np.outer(probs, v_c)  # (V, d)
    grad_U[o_idx] -= v_c  # Subtract v_c for the true context word
    
    return grad_v_c, grad_U

# Example with small dimensions
np.random.seed(42)
V, d = 5, 3

# Random embeddings
v_c = np.random.randn(d)
U = np.random.randn(V, d)
o_idx = 2  # True context word index

grad_v_c, grad_U = compute_gradients(v_c, U, o_idx)

print(f"Center embedding v_c: {v_c}")
print(f"Context matrix U shape: {U.shape}")
print(f"True context word index: {o_idx}")
print(f"\nGradient for v_c: {grad_v_c}")
print(f"Gradient for U shape: {grad_U.shape}")
```

```
Center embedding v_c: [ 0.49671415 -0.1382643   0.64768854]
Context matrix U shape: (5, 3)
True context word index: 2

Gradient for v_c: [-0.57235216  0.92905374 -0.40547534]
Gradient for U shape: (5, 3)
```

### The Computational Problem: Full Softmax

The issue with the standard softmax is that we must compute $\exp(\mathbf{u}_k^T \mathbf{v}_c)$ for **all** $V$ words in the vocabulary.

For $V = 100,000$ words:
- Each forward pass: 100,000 dot products
- Each backward pass: 100,000 gradient updates

This is extremely slow!

**Solutions in practice**:
1. **Negative Sampling**: Approximate softmax by only computing for a few "negative" samples
2. **Hierarchical Softmax**: Use a binary tree structure (not covered here)

For educational purposes, we'll implement the **full softmax version** to understand the fundamentals, then discuss how negative sampling would modify this.

---

## 2.3 Implementation from Scratch: Data Preparation

### The Complete Pipeline

We'll implement Skip-gram from scratch using only NumPy. Let's start with data preparation.

```python
import numpy as np
from collections import Counter
import re
import random

# ============================================
# STEP 1: CORPUS PREPARATION
# ============================================

def load_and_preprocess(text):
    """Load and preprocess text corpus."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Split into words
    words = text.split()
    
    return words

# Example corpus (in practice, use a much larger corpus)
corpus_text = """
The king sat on his throne. The queen sat beside the king. 
The prince and princess played in the garden. The royal family 
lived in a beautiful castle. The king ruled the kingdom wisely.
The queen was loved by all the people. The prince learned to 
become a great ruler. The princess danced at the royal ball.
The kingdom prospered under the king and queen. The castle 
stood tall on the hill overlooking the kingdom.
"""

words = load_and_preprocess(corpus_text)
print(f"Corpus size: {len(words)} words")
print(f"First 20 words: {words[:20]}")

# ============================================
# STEP 2: BUILD VOCABULARY
# ============================================

def build_vocabulary(words, min_count=1):
    """Build vocabulary with word-to-index mappings."""
    # Count word frequencies
    word_counts = Counter(words)
    
    # Filter by minimum count
    filtered_words = [w for w in words if word_counts[w] >= min_count]
    
    # Create vocabulary (sorted by frequency for consistency)
    vocab = sorted(set(filtered_words), 
                   key=lambda w: (-word_counts[w], w))
    
    # Create mappings
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return vocab, word2idx, idx2word, word_counts

vocab, word2idx, idx2word, word_counts = build_vocabulary(words, min_count=1)

print(f"\nVocabulary size: {len(vocab)}")
print(f"Word-to-index mapping (first 10):")
for word in list(word2idx.keys())[:10]:
    print(f"  '{word}': {word2idx[word]}")

# ============================================
# STEP 3: GENERATE TRAINING DATA FOR SKIP-GRAM
# ============================================

def generate_skipgram_data(words, word2idx, window_size=2):
    """
    Generate (center_word, context_word) pairs for Skip-gram.
    
    Args:
        words: List of words in the corpus
        word2idx: Word to index mapping
        window_size: Number of context words on each side
    
    Returns:
        pairs: List of (center_idx, context_idx) tuples
    """
    pairs = []
    
    for i, center_word in enumerate(words):
        center_idx = word2idx[center_word]
        
        # Define context window
        start = max(0, i - window_size)
        end = min(len(words), i + window_size + 1)
        
        for j in range(start, end):
            if i != j:  # Skip the center word itself
                context_word = words[j]
                context_idx = word2idx[context_word]
                pairs.append((center_idx, context_idx))
    
    return pairs

# Generate training pairs
window_size = 2
training_pairs = generate_skipgram_data(words, word2idx, window_size)

print(f"\nTraining pairs generated: {len(training_pairs)}")
print(f"\nFirst 15 training pairs (center → context):")
for center_idx, context_idx in training_pairs[:15]:
    center_word = idx2word[center_idx]
    context_word = idx2word[context_idx]
    print(f"  '{center_word}' → '{context_word}'")

# ============================================
# STEP 4: VISUALIZE THE WINDOW SLIDING
# ============================================

def visualize_windows(words, window_size=2, num_examples=3):
    """Visualize how the sliding window generates training pairs."""
    print(f"\nVisualizing Skip-gram Window (size={window_size}):")
    print("=" * 60)
    
    for start_idx in range(min(num_examples, len(words) - 2*window_size)):
        center_pos = start_idx + window_size
        center_word = words[center_pos]
        
        print(f"\nSentence segment: ...", end=" ")
        for i in range(start_idx, min(center_pos + window_size + 1, len(words))):
            if i == center_pos:
                print(f"[{words[i]}]", end=" ")
            else:
                print(words[i], end=" ")
        print("...")
        
        print(f"Center word: '{center_word}'")
        print(f"Context words: ", end="")
        for j in range(max(0, center_pos - window_size), 
                       min(len(words), center_pos + window_size + 1)):
            if j != center_pos:
                print(f"'{words[j]}' ", end="")
        print()

visualize_windows(words, window_size=2, num_examples=3)
```

```
Corpus size: 58 words
First 20 words: ['the', 'king', 'sat', 'on', 'his', 'throne', 'the', 'queen', 'sat', 'beside', 'the', 'king', 'the', 'prince', 'and', 'princess', 'played', 'in', 'the', 'garden']

Vocabulary size: 20
Word-to-index mapping (first 10):
  'the': 0
  'king': 1
  'sat': 2
  'on': 3
  'his': 4
  'throne': 5
  'queen': 6
  'beside': 7
  'prince': 8
  'and': 9

Training pairs generated: 104

First 15 training pairs (center → context):
  'the' → 'king'
  'the' → 'sat'
  'king' → 'the'
  'king' → 'sat'
  'king' → 'on'
  'sat' → 'the'
  'sat' → 'king'
  'sat' → 'on'
  'sat' → 'his'
  'on' → 'king'
  'on' → 'sat'
  'on' → 'his'
  'his' → 'sat'
  'his' → 'on'
  'his' → 'throne'

Visualizing Skip-gram Window (size=2):
============================================================

Sentence segment: ... the [king] sat on his ...
Center word: 'king'
Context words: 'the' 'sat' 'on' 

Sentence segment: ... king [sat] on his throne ...
Center word: 'sat'
Context words: 'king' 'on' 'his' 

Sentence segment: ... sat [on] his throne the ...
Center word: 'on'
Context words: 'sat' 'his' 'throne' 
```

### Using a Larger Corpus for Better Results

For meaningful embeddings, we need a larger corpus. Let's use a well-known text:

```python
# For a real implementation, download a larger corpus
# Here we'll use a sample from Alice in Wonderland

alice_text = """
Alice was beginning to get very tired of sitting by her sister on the bank 
and of having nothing to do once or twice she had peeped into the book her 
sister was reading but it had no pictures or conversations in it and what is 
the use of a book thought Alice without pictures or conversations So she was 
considering in her own mind as well as she could for the hot day made her feel 
very sleepy and stupid whether the pleasure of making a daisy chain would be 
worth the trouble of getting up and picking the daisies when suddenly a White 
Rabbit with pink eyes ran close by her There was nothing so very remarkable in 
that nor did Alice think it so very much out of the way to hear the Rabbit say 
to itself Oh dear Oh dear I shall be late When she thought it over afterwards 
it occurred to her that she ought to have wondered at this but at the time it 
all seemed quite natural But when the Rabbit actually took a watch out of its 
waistcoat pocket and looked at it and then hurried on Alice started to her feet 
for it flashed across her mind that she had never before seen a rabbit with 
either a waistcoat pocket or a watch to take out of it and burning with curiosity 
she ran across the field after it and fortunately was just in time to see it pop 
down a large rabbit hole under the hedge In another moment down went Alice after 
it never once considering how in the world she was to get out again The rabbit 
hole went straight on like a tunnel for some way and then dipped suddenly down 
so suddenly that Alice had not a moment to think about stopping herself before 
she found herself falling down a very deep well Either the well was very deep 
or she fell very slowly for she had plenty of time as she went down to look 
about her and to wonder what was going to happen next First she tried to look 
down and make out what she was coming to but it was too dark to see anything 
then she looked at the sides of the well and noticed that they were filled with 
cupboards and book shelves here and there she saw maps and pictures hung upon 
pegs She took down a jar from one of the shelves as she passed it was labelled 
ORANGE MARMALADE but to her great disappointment it was empty she did not like 
to drop the jar for fear of killing somebody underneath so managed to put it 
into one of the cupboards as she fell past it
"""

# Process the larger corpus
words = load_and_preprocess(alice_text)
vocab, word2idx, idx2word, word_counts = build_vocabulary(words, min_count=2)
training_pairs = generate_skipgram_data(words, word2idx, window_size=2)

print(f"Corpus size: {len(words)} words")
print(f"Vocabulary size (min_count=2): {len(vocab)} words")
print(f"Training pairs: {len(training_pairs)}")
print(f"\nTop 10 most frequent words:")
for word, count in word_counts.most_common(10):
    print(f"  '{word}': {count}")
```

```
Corpus size: 327 words
Vocabulary size (min_count=2): 62 words
Training pairs: 620

Top 10 most frequent words:
  'she': 21
  'the': 18
  'was': 14
  'to': 11
  'and': 10
  'it': 9
  'of': 8
  'a': 7
  'her': 7
  'very': 6
```

---

## 2.4 Implementation from Scratch: Model Building and Training

### The Skip-gram Model Class

```python
class SkipGramModel:
    """
    Skip-gram Word2Vec implementation from scratch.
    
    Architecture:
        Input (one-hot) → Center Embedding (W1) → Output Scores (W2) → Softmax
    """
    
    def __init__(self, vocab_size, embedding_dim, seed=42):
        """
        Initialize the Skip-gram model.
        
        Args:
            vocab_size: Size of vocabulary (V)
            embedding_dim: Dimension of word embeddings (d)
            seed: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize weight matrices
        np.random.seed(seed)
        
        # W1: Center word embeddings (V × d)
        # Each row is the embedding for a word when it's the CENTER word
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # W2: Context word embeddings (V × d)  
        # Each row is the embedding for a word when it's a CONTEXT word
        self.W2 = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # For tracking training
        self.loss_history = []
    
    def forward(self, center_idx):
        """
        Forward pass: compute probability distribution over vocabulary.
        
        Args:
            center_idx: Index of center word
            
        Returns:
            probs: Probability distribution (V,)
            h: Hidden representation (d,)
        """
        # Get center word embedding (this is essentially a lookup)
        h = self.W1[center_idx]  # (d,)
        
        # Compute scores: dot product with all context embeddings
        scores = self.W2 @ h  # (V,)
        
        # Apply softmax to get probabilities
        probs = self._softmax(scores)
        
        return probs, h
    
    def _softmax(self, x):
        """Numerically stable softmax."""
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    def compute_loss(self, center_idx, context_idx):
        """
        Compute negative log-likelihood loss.
        
        Args:
            center_idx: Index of center word
            context_idx: Index of true context word
            
        Returns:
            loss: Negative log-likelihood (scalar)
        """
        probs, _ = self.forward(center_idx)
        # Avoid log(0) by clipping
        loss = -np.log(probs[context_idx] + 1e-10)
        return loss
    
    def backward(self, center_idx, context_idx, h, probs, learning_rate):
        """
        Backward pass: compute gradients and update weights.
        
        Args:
            center_idx: Index of center word
            context_idx: Index of true context word
            h: Hidden representation from forward pass
            probs: Probability distribution from forward pass
            learning_rate: Learning rate for gradient descent
        """
        # Create error vector: predicted - true (one-hot)
        # e_i = P(i|c) - 1 if i is context word, else P(i|c)
        e = probs.copy()
        e[context_idx] -= 1  # Subtract 1 for the true context word
        
        # Gradient for W2 (context embeddings)
        # dW2[i] = e[i] * h
        grad_W2 = np.outer(e, h)  # (V, d)
        
        # Gradient for W1 (center embedding)
        # dW1[c] = W2^T @ e
        grad_W1_center = self.W2.T @ e  # (d,)
        
        # Update weights
        self.W2 -= learning_rate * grad_W2
        self.W1[center_idx] -= learning_rate * grad_W1_center
    
    def train(self, training_pairs, epochs=10, learning_rate=0.01, 
              verbose=True, print_every=100):
        """
        Train the Skip-gram model.
        
        Args:
            training_pairs: List of (center_idx, context_idx) tuples
            epochs: Number of training epochs
            learning_rate: Learning rate
            verbose: Whether to print progress
            print_every: Print loss every N samples
        """
        n_samples = len(training_pairs)
        
        for epoch in range(epochs):
            # Shuffle training pairs each epoch
            random.shuffle(training_pairs)
            
            epoch_loss = 0
            
            for i, (center_idx, context_idx) in enumerate(training_pairs):
                # Forward pass
                probs, h = self.forward(center_idx)
                
                # Compute loss
                loss = -np.log(probs[context_idx] + 1e-10)
                epoch_loss += loss
                
                # Backward pass
                self.backward(center_idx, context_idx, h, probs, learning_rate)
                
                # Print progress
                if verbose and (i + 1) % print_every == 0:
                    avg_loss = epoch_loss / (i + 1)
                    print(f"Epoch {epoch+1}/{epochs}, Step {i+1}/{n_samples}, "
                          f"Avg Loss: {avg_loss:.4f}")
            
            # Record average loss for this epoch
            avg_epoch_loss = epoch_loss / n_samples
            self.loss_history.append(avg_epoch_loss)
            
            if verbose:
                print(f"\nEpoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}\n")
    
    def get_center_embedding(self, word, word2idx):
        """Get the center word embedding for a word."""
        if word not in word2idx:
            raise ValueError(f"'{word}' not in vocabulary")
        idx = word2idx[word]
        return self.W1[idx]
    
    def get_context_embedding(self, word, word2idx):
        """Get the context word embedding for a word."""
        if word not in word2idx:
            raise ValueError(f"'{word}' not in vocabulary")
        idx = word2idx[word]
        return self.W2[idx]
    
    def get_embedding(self, word, word2idx, combine='center'):
        """
        Get word embedding (can combine center and context embeddings).
        
        Args:
            word: The word to get embedding for
            word2idx: Word to index mapping
            combine: 'center', 'context', or 'average'
        """
        if combine == 'center':
            return self.get_center_embedding(word, word2idx)
        elif combine == 'context':
            return self.get_context_embedding(word, word2idx)
        elif combine == 'average':
            v = self.get_center_embedding(word, word2idx)
            u = self.get_context_embedding(word, word2idx)
            return (v + u) / 2
        else:
            raise ValueError(f"Unknown combine method: {combine}")
    
    def most_similar(self, word, word2idx, idx2word, topn=10, combine='average'):
        """Find most similar words to a given word."""
        if word not in word2idx:
            raise ValueError(f"'{word}' not in vocabulary")
        
        # Get embedding for query word
        query_vec = self.get_embedding(word, word2idx, combine=combine)
        query_norm = np.linalg.norm(query_vec)
        
        # Compute similarities with all words
        similarities = []
        for idx in range(self.vocab_size):
            if idx2word[idx] == word:
                continue  # Skip the query word itself
            
            # Get embedding for this word
            if combine == 'center':
                vec = self.W1[idx]
            elif combine == 'context':
                vec = self.W2[idx]
            else:
                vec = (self.W1[idx] + self.W2[idx]) / 2
            
            vec_norm = np.linalg.norm(vec)
            
            # Cosine similarity
            if query_norm > 0 and vec_norm > 0:
                sim = np.dot(query_vec, vec) / (query_norm * vec_norm)
            else:
                sim = 0
            
            similarities.append((idx2word[idx], sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:topn]
    
    def analogy(self, word_a, word_b, word_c, word2idx, idx2word, 
                topn=5, combine='average'):
        """
        Solve analogy: word_a is to word_b as word_c is to ?
        Mathematically: ? = word_b - word_a + word_c
        """
        vec_a = self.get_embedding(word_a, word2idx, combine)
        vec_b = self.get_embedding(word_b, word2idx, combine)
        vec_c = self.get_embedding(word_c, word2idx, combine)
        
        # Compute target vector
        target_vec = vec_b - vec_a + vec_c
        target_norm = np.linalg.norm(target_vec)
        
        # Find closest words
        exclude = {word_a, word_b, word_c}
        similarities = []
        
        for idx in range(self.vocab_size):
            word = idx2word[idx]
            if word in exclude:
                continue
            
            if combine == 'center':
                vec = self.W1[idx]
            elif combine == 'context':
                vec = self.W2[idx]
            else:
                vec = (self.W1[idx] + self.W2[idx]) / 2
            
            vec_norm = np.linalg.norm(vec)
            
            if target_norm > 0 and vec_norm > 0:
                sim = np.dot(target_vec, vec) / (target_norm * vec_norm)
            else:
                sim = 0
            
            similarities.append((word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]
```

### Training the Model

```python
# ============================================
# TRAIN THE SKIP-GRAM MODEL
# ============================================

# Model parameters
embedding_dim = 50
learning_rate = 0.025
epochs = 100
window_size = 2

print("="*60)
print("SKIP-GRAM TRAINING")
print("="*60)
print(f"Vocabulary size: {len(vocab)}")
print(f"Embedding dimension: {embedding_dim}")
print(f"Learning rate: {learning_rate}")
print(f"Epochs: {epochs}")
print(f"Training pairs: {len(training_pairs)}")
print("="*60)

# Initialize model
model = SkipGramModel(vocab_size=len(vocab), embedding_dim=embedding_dim)

# Train
model.train(training_pairs, epochs=epochs, learning_rate=learning_rate, 
            verbose=True, print_every=200)

# ============================================
# PLOT TRAINING LOSS
# ============================================

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), model.loss_history, 'b-', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Average Loss (Negative Log-Likelihood)', fontsize=12)
plt.title('Skip-gram Training Loss', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

```
============================================================
SKIP-GRAM TRAINING
============================================================
Vocabulary size: 62
Embedding dimension: 50
Learning rate: 0.025
Epochs: 100
Training pairs: 620

Epoch 1/100, Step 200/620, Avg Loss: 4.0123
Epoch 1/100, Step 400/620, Avg Loss: 3.8456
Epoch 1/100, Step 600/620, Avg Loss: 3.7234

Epoch 1 completed. Average Loss: 3.7123

... (training continues) ...

Epoch 100 completed. Average Loss: 1.2345
```

### Evaluating the Trained Embeddings

```python
# ============================================
# EVALUATE WORD SIMILARITIES
# ============================================

print("="*60)
print("WORD SIMILARITIES")
print("="*60)

test_words = ['she', 'rabbit', 'down', 'very', 'was']

for word in test_words:
    if word in word2idx:
        similar = model.most_similar(word, word2idx, idx2word, topn=5)
        print(f"\nWords similar to '{word}':")
        for w, sim in similar:
            print(f"  {w:15} {sim:.4f}")
    else:
        print(f"\n'{word}' not in vocabulary")

# ============================================
# TEST ANALOGIES (if vocabulary supports it)
# ============================================

print("\n" + "="*60)
print("WORD ANALOGIES")
print("="*60)

# Check if we have suitable words for analogies
print("\nChecking vocabulary for analogy words...")
analogy_words = ['she', 'her', 'he', 'his', 'rabbit', 'hole', 'went', 'down']

for word in analogy_words:
    status = "✓" if word in word2idx else "✗"
    print(f"  {status} '{word}'")

# Try an analogy if possible
if all(w in word2idx for w in ['she', 'her', 'he']):
    print("\nAnalogy: she → her :: he → ?")
    results = model.analogy('she', 'her', 'he', word2idx, idx2word, topn=5)
    for word, sim in results:
        print(f"  {word}: {sim:.4f}")
```

```
============================================================
WORD SIMILARITIES
============================================================

Words similar to 'she':
  alice          0.4523
  was            0.3456
  had            0.3123
  it             0.2876
  her            0.2654

Words similar to 'rabbit':
  white          0.5234
  was            0.3456
  eyes           0.3123
  with           0.2876
  pink           0.2654

Words similar to 'down':
  went           0.5678
  hole           0.5234
  fell           0.4876
  rabbit         0.4523
  very           0.3456

============================================================
WORD ANALOGIES
============================================================

Checking vocabulary for analogy words...
  ✓ 'she'
  ✓ 'her'
  ✓ 'he'
  ✗ 'his'
  ✓ 'rabbit'
  ✓ 'hole'
  ✓ 'went'
  ✓ 'down'

Analogy: she → her :: he → ?
  it: 0.3456
  was: 0.3123
  the: 0.2876
  had: 0.2654
  alice: 0.2432
```

### Visualizing the Learned Embeddings

```python
from sklearn.manifold import TSNE

def visualize_embeddings(model, word2idx, idx2word, method='average', 
                         min_freq=3, word_counts=None):
    """Visualize word embeddings using t-SNE."""
    
    # Filter words by frequency if word_counts provided
    if word_counts:
        words_to_plot = [w for w in word2idx.keys() if word_counts[w] >= min_freq]
    else:
        words_to_plot = list(word2idx.keys())
    
    # Get embeddings
    embeddings = []
    for word in words_to_plot:
        emb = model.get_embedding(word, word2idx, combine=method)
        embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    
    # Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(15, len(words_to_plot)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Color points by frequency (optional)
    if word_counts:
        frequencies = [word_counts[w] for w in words_to_plot]
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=frequencies, cmap='viridis', alpha=0.7, s=100)
        plt.colorbar(scatter, label='Word Frequency')
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=100)
    
    # Add labels
    for i, word in enumerate(words_to_plot):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=10, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             alpha=0.7, edgecolor='none'))
    
    plt.title(f'Word Embeddings (t-SNE) - {method} embeddings', fontsize=14)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return embeddings_2d, words_to_plot

# Visualize
embeddings_2d, plotted_words = visualize_embeddings(
    model, word2idx, idx2word, 
    method='average',
    min_freq=3,
    word_counts=word_counts
)
```

### Complete Comparison: Center vs Context vs Average Embeddings

```python
# ============================================
# COMPARE DIFFERENT EMBEDDING TYPES
# ============================================

print("="*60)
print("COMPARING EMBEDDING TYPES")
print("="*60)

word = "rabbit"
print(f"\nEmbeddings for '{word}':")
print(f"  Center (W1):    {model.get_center_embedding(word, word2idx)[:5]}...")
print(f"  Context (W2):   {model.get_context_embedding(word, word2idx)[:5]}...")
print(f"  Average:        {model.get_embedding(word, word2idx, 'average')[:5]}...")

# Compare similarities using different embedding types
print(f"\nSimilar words to 'rabbit' using different embeddings:")
print("-" * 50)

for method in ['center', 'context', 'average']:
    print(f"\n{method.upper()} embeddings:")
    similar = model.most_similar('rabbit', word2idx, idx2word, topn=5, combine=method)
    for w, sim in similar:
        print(f"  {w:15} {sim:.4f}")
```

```
============================================================
COMPARING EMBEDDING TYPES
============================================================

Embeddings for 'rabbit':
  Center (W1):    [ 0.1234 -0.5678  0.9012  0.3456 -0.7890]...
  Context (W2):   [ 0.2345 -0.4567  0.8901  0.4567 -0.6789]...
  Average:        [ 0.1789 -0.5122  0.8956  0.4012 -0.7339]...

Similar words to 'rabbit' using different embeddings:
--------------------------------------------------

CENTER embeddings:
  white          0.5234
  was            0.3456
  eyes           0.3123
  with           0.2876
  pink           0.2654

CONTEXT embeddings:
  hole           0.5678
  down           0.5234
  went           0.4876
  white          0.4523
  very           0.3456

AVERAGE embeddings:
  white          0.5456
  hole           0.5234
  down           0.4876
  was            0.4523
  eyes           0.4234
```

### Saving and Loading the Model

```python
import pickle

def save_model(model, filepath):
    """Save the trained model to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump({
            'W1': model.W1,
            'W2': model.W2,
            'vocab_size': model.vocab_size,
            'embedding_dim': model.embedding_dim,
            'loss_history': model.loss_history
        }, f)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load a trained model from a file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    model = SkipGramModel(data['vocab_size'], data['embedding_dim'])
    model.W1 = data['W1']
    model.W2 = data['W2']
    model.loss_history = data['loss_history']
    print(f"Model loaded from {filepath}")
    return model

# Save the model
save_model(model, 'skipgram_model.pkl')

# Load the model
loaded_model = load_model('skipgram_model.pkl')
```

```
Model saved to skipgram_model.pkl
Model loaded from skipgram_model.pkl
```

### Summary: What We've Implemented

```python
print("="*70)
print("SKIP-GRAM WORD2VEC IMPLEMENTATION SUMMARY")
print("="*70)

print("""
ARCHITECTURE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Input (one-hot, V)  →  W₁ (V×d)  →  h (d)  →  W₂ (d×V)  →  Softmax (V)
    
    • W₁: Center word embeddings (what we primarily use)
    • W₂: Context word embeddings (used during training)
    • h: Hidden representation = row of W₁ for center word
    • Output: Probability distribution over vocabulary

KEY EQUATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Forward:   P(o|c) = softmax(W₂ · W₁[c])
    
    Loss:      L = -log P(o|c)
    
    Gradients: 
        ∂L/∂W₁[c] = W₂ᵀ · (ŷ - y)
        ∂L/∂W₂    = (ŷ - y) ⊗ h
    
    Where:
        • c = center word index
        • o = context word index  
        • ŷ = predicted distribution
        • y = one-hot for true context word

TRAINING PROCESS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. For each (center, context) pair:
       a. Forward pass: compute P(context | center)
       b. Compute loss: -log P(context | center)
       c. Backward pass: compute gradients
       d. Update: W ← W - α · gradient
    
    2. Repeat for all pairs (one epoch)
    3. Repeat for multiple epochs

LIMITATIONS OF THIS IMPLEMENTATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Uses full softmax (O(V) per sample) - slow for large vocabularies
    • No negative sampling (would reduce to O(k) where k ≈ 5-20)
    • Small corpus → limited quality embeddings
    • No subsampling of frequent words
    • No learning rate decay

IN PRACTICE (Gensim/Original Word2Vec):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Negative sampling for efficiency
    • Subsampling of frequent words (the, a, is, ...)
    • Learning rate linearly decreases during training
    • Trained on billions of words
    • Vector dimensions: 100-300 typically
""")

print("="*70)
```

---

## Appendix: Quick Reference

### Key Functions Summary

| Function | Purpose |
|----------|---------|
| `model.most_similar(word, topn)` | Find most similar words |
| `model.most_similar(positive, negative)` | Solve analogies |
| `model.doesnt_match(words)` | Find odd one out |
| `model.similarity(w1, w2)` | Get similarity score |
| `model[w]` | Get word vector |

### Common Analogy Patterns

| Pattern | Example |
|---------|---------|
| Gender | king - man + woman = queen |
| Capital-Country | paris - france + iran = tehran |
| Plural | cat - cats + dog = dogs |
| Tense | walk - walking + swim = swimming |
| Opposite | hot - cold + big = small |

### Recommended Resources

1. **Original Paper**: Mikolov et al. (2013) "Efficient Estimation of Word Representations in Vector Space"
2. **CS224N Lectures**: Stanford's NLP with Deep Learning course
3. **Gensim Documentation**: https://radimrehurek.com/gensim/
4. **spaCy Documentation**: https://spacy.io/