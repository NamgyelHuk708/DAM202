# Luong Attention Mechanism: Complete Numerical Example

## Overview
This report demonstrates how the Luong Attention mechanism works using a concrete numerical example with real matrix calculations.

---

## Example Sentence:
- **English Input**: "I love you"  
- **French Output**: "Je t'aime" (generating "t'aime" given "Je" was already generated)

---

## Setup Parameters

```python
# Simplified dimensions for clarity
vocab_size_en = 1000
vocab_size_fr = 1200
embedding_dim = 4
enc_units = 3
dec_units = 3
batch_size = 1
max_length = 3
```

---

## STEP 1: Input Encoding

### Input Sentence: "I love you"
After tokenization and embedding:

```
Word IDs: [45, 123, 78]

Embedded Input (after embedding layer):
X = [[0.5, 0.2, -0.3, 0.1],    # "I"
     [0.8, -0.1, 0.4, 0.2],    # "love"
     [-0.2, 0.6, 0.3, -0.5]]   # "you"

Shape: (3, 4) = (sequence_length, embedding_dim)
```

---

## STEP 2: Encoder Output (After Bidirectional LSTM)

The LSTM processes the embedded input and produces:

```
Encoder Outputs (H):
H = [[0.3, 0.7, -0.2],    # Hidden state for "I"
     [0.5, 0.4, 0.6],     # Hidden state for "love"
     [0.2, -0.3, 0.8]]    # Hidden state for "you"

Shape: (3, 3) = (max_length, enc_units)

Last encoder state (used to initialize decoder):
h_enc = [0.2, -0.3, 0.8]
c_enc = [0.1, 0.5, -0.2]
```

---

## STEP 3: Decoder at Time Step t=1

**Already Generated:** "Je" (French for "I")  
**Now Generating:** Next word (should be "t'aime")

### Decoder Input:
```
Previous word embedding for "Je":
x_dec = [0.4, -0.2, 0.3, 0.1]
Shape: (1, 4)
```

### Decoder LSTM Output:
```
After LSTM processing:
decoder_hidden (query) = [0.6, -0.4, 0.5]
Shape: (1, 3) = (batch_size, dec_units)
```

---

## STEP 4: ATTENTION MECHANISM (The Core Part!)

### 4.1: Transform Encoder Outputs

**Weight Matrix W (learnable):**
```
W = [[0.8, -0.2, 0.3],
     [0.5, 0.6, -0.1],
     [-0.3, 0.4, 0.7]]

Shape: (enc_units=3, dec_units=3)
```

**Apply transformation:**
```
H_transformed = H · W

Step-by-step calculation:
Row 1: [0.3, 0.7, -0.2] · W = [0.3×0.8 + 0.7×0.5 + (-0.2)×(-0.3),  ...]
                              = [0.65, 0.22, 0.02]

Row 2: [0.5, 0.4, 0.6] · W  = [0.76, 0.22, 0.53]

Row 3: [0.2, -0.3, 0.8] · W = [0.41, -0.06, 0.59]

Result:
H_transformed = [[0.65, 0.22, 0.02],
                 [0.76, 0.22, 0.53],
                 [0.41, -0.06, 0.59]]

Shape: (3, 3)
```

---

### 4.2: Calculate Attention Scores

**Query (decoder hidden state):**
```
q = [0.6, -0.4, 0.5]
Shape: (1, 3)
```

**Expand query for broadcasting:**
```
q_expanded = [[0.6, -0.4, 0.5]]
Shape: (1, 1, 3)
```

**Compute scores (dot product with each encoder output):**
```
score[i] = q · H_transformed[i]

score[0] = [0.6, -0.4, 0.5] · [0.65, 0.22, 0.02]
         = 0.6×0.65 + (-0.4)×0.22 + 0.5×0.02
         = 0.39 - 0.088 + 0.01
         = 0.312

score[1] = [0.6, -0.4, 0.5] · [0.76, 0.22, 0.53]
         = 0.6×0.76 + (-0.4)×0.22 + 0.5×0.53
         = 0.456 - 0.088 + 0.265
         = 0.633

score[2] = [0.6, -0.4, 0.5] · [0.41, -0.06, 0.59]
         = 0.6×0.41 + (-0.4)×(-0.06) + 0.5×0.59
         = 0.246 + 0.024 + 0.295
         = 0.565

Scores = [0.312, 0.633, 0.565]
Shape: (3,)
```

**Interpretation:** 
- Word "love" (score=0.633) gets highest attention
- Word "you" (score=0.565) gets second highest
- Word "I" (score=0.312) gets lowest

---

### 4.3: Apply Softmax to Get Attention Weights

```
Softmax formula: α[i] = exp(score[i]) / Σ(exp(score[j]))

exp(scores) = [exp(0.312), exp(0.633), exp(0.565)]
            = [1.366, 1.883, 1.760]

Sum = 1.366 + 1.883 + 1.760 = 5.009

Attention weights (α):
α = [1.366/5.009, 1.883/5.009, 1.760/5.009]
  = [0.273, 0.376, 0.351]

Shape: (3,)
```

**Interpretation:**
- 27.3% attention on "I"
- **37.6% attention on "love"** ← Highest focus!
- 35.1% attention on "you"

**This makes sense!** When generating "t'aime" (love), the model focuses most on "love" in the input.

---

### 4.4: Calculate Context Vector

**Weighted sum of encoder outputs:**
```
Context vector = Σ(α[i] × H[i])

c = 0.273 × [0.3, 0.7, -0.2] + 
    0.376 × [0.5, 0.4, 0.6] + 
    0.351 × [0.2, -0.3, 0.8]

Calculation:
Dimension 1: 0.273×0.3 + 0.376×0.5 + 0.351×0.2 = 0.082 + 0.188 + 0.070 = 0.340
Dimension 2: 0.273×0.7 + 0.376×0.4 + 0.351×(-0.3) = 0.191 + 0.150 - 0.105 = 0.236
Dimension 3: 0.273×(-0.2) + 0.376×0.6 + 0.351×0.8 = -0.055 + 0.226 + 0.281 = 0.452

Context vector:
c = [0.340, 0.236, 0.452]
Shape: (1, 3)
```

---

## STEP 5: Combine Context with Decoder Output

```
Decoder hidden state: h_dec = [0.6, -0.4, 0.5]
Context vector:       c     = [0.340, 0.236, 0.452]

Concatenate:
combined = [0.6, -0.4, 0.5, 0.340, 0.236, 0.452]
Shape: (1, 6)
```

---

## STEP 6: Final Transformation

### Apply tanh transformation:
```
Weight matrix Wc:
Wc = [[0.5, -0.3, 0.2, 0.1, 0.4, -0.2],
      [0.3, 0.6, -0.1, 0.5, -0.3, 0.2],
      [-0.2, 0.4, 0.7, -0.3, 0.1, 0.5]]

Shape: (3, 6)

output = Wc · combined^T
       = [...calculation...]
       = [0.345, -0.123, 0.567]

Apply tanh:
output_tanh = tanh([0.345, -0.123, 0.567])
            = [0.332, -0.122, 0.513]

Shape: (1, 3)
```

### Final prediction layer:
```
Weight matrix Wo (maps to vocabulary):
Wo = random matrix of shape (vocab_size_fr=1200, 3)

Logits = Wo · output_tanh^T
Shape: (1, 1200)

After softmax:
probabilities = softmax(logits)

Example output (top 3 words):
Word ID 567 ("t'aime"): 0.342  ← Highest probability!
Word ID 234 ("adore"):  0.198
Word ID 891 ("aime"):   0.156
...
```

**Predicted word:** "t'aime"

---

## Summary of the Complete Flow

```
INPUT: "I love you" → Embedded → LSTM Encoder
                                      ↓
                              Encoder Outputs H
                              [3 hidden states]
                                      ↓
DECODER: "Je" → Embedded → LSTM → decoder_hidden (query)
                                      ↓
                        ┌─────────────┴─────────────┐
                        │   ATTENTION MECHANISM      │
                        │                            │
                        │  1. Transform H with W     │
                        │  2. Score = query · H_tr   │
                        │  3. Softmax → weights α    │
                        │     [0.273, 0.376, 0.351]  │
                        │  4. Context = Σ(α × H)     │
                        └─────────────┬─────────────┘
                                      ↓
                        Concat [context, decoder_out]
                                      ↓
                              tanh(Wc · concat)
                                      ↓
                              softmax(Wo · ...)
                                      ↓
                        OUTPUT: "t'aime" (love)
```

---

## Key Characteristics of Luong Attention

1. **"General" scoring function**: Uses learnable weight matrix W
   - `score(h_t, h_s) = h_t^T · W · h_s`

2. **Post-LSTM attention**: Attention is calculated AFTER the decoder LSTM processes the input

3. **Simpler than Bahdanau**: 
   - Bahdanau uses concatenation + tanh + another layer
   - Luong directly uses dot product (faster computation)

4. **Alignment**: Shows which source words the model focuses on when generating each target word

---

## Why Luong Attention is Effective

- Dynamic Focus**: Different attention weights for each decoding step  
- Handles Variable Lengths**: Works with any input/output sequence length  
- Interpretable**: Attention weights visualize what the model "looks at"  
- Gradient Flow**: Provides direct path for gradients from decoder to encoder  

---

## Conclusion
# Luong Attention Mechanism: Complete Numerical Example

## Overview
This report demonstrates how the Luong Attention mechanism works using a concrete numerical example with real matrix calculations.

---

## Example Sentence:
- **English Input**: "I love you"  
- **French Output**: "Je t'aime" (generating "t'aime" given "Je" was already generated)

---

## Setup Parameters

```python
# Simplified dimensions for clarity
vocab_size_en = 1000
vocab_size_fr = 1200
embedding_dim = 4
enc_units = 3
dec_units = 3
batch_size = 1
max_length = 3
```

---

## STEP 1: Input Encoding

### Input Sentence: "I love you"
After tokenization and embedding:

```
Word IDs: [45, 123, 78]

Embedded Input (after embedding layer):
X = [[0.5, 0.2, -0.3, 0.1],    # "I"
     [0.8, -0.1, 0.4, 0.2],    # "love"
     [-0.2, 0.6, 0.3, -0.5]]   # "you"

Shape: (3, 4) = (sequence_length, embedding_dim)
```

---

## STEP 2: Encoder Output (After Bidirectional LSTM)

The LSTM processes the embedded input and produces:

```
Encoder Outputs (H):
H = [[0.3, 0.7, -0.2],    # Hidden state for "I"
     [0.5, 0.4, 0.6],     # Hidden state for "love"
     [0.2, -0.3, 0.8]]    # Hidden state for "you"

Shape: (3, 3) = (max_length, enc_units)

Last encoder state (used to initialize decoder):
h_enc = [0.2, -0.3, 0.8]
c_enc = [0.1, 0.5, -0.2]
```

---

## STEP 3: Decoder at Time Step t=1

**Already Generated:** "Je" (French for "I")  
**Now Generating:** Next word (should be "t'aime")

### Decoder Input:
```
Previous word embedding for "Je":
x_dec = [0.4, -0.2, 0.3, 0.1]
Shape: (1, 4)
```

### Decoder LSTM Output:
```
After LSTM processing:
decoder_hidden (query) = [0.6, -0.4, 0.5]
Shape: (1, 3) = (batch_size, dec_units)
```

---

## STEP 4: ATTENTION MECHANISM (The Core Part!)

### 4.1: Transform Encoder Outputs

**Weight Matrix W (learnable):**
```
W = [[0.8, -0.2, 0.3],
     [0.5, 0.6, -0.1],
     [-0.3, 0.4, 0.7]]

Shape: (enc_units=3, dec_units=3)
```

**Apply transformation:**
```
H_transformed = H · W

Step-by-step calculation:
Row 1: [0.3, 0.7, -0.2] · W = [0.3×0.8 + 0.7×0.5 + (-0.2)×(-0.3),  ...]
                              = [0.65, 0.22, 0.02]

Row 2: [0.5, 0.4, 0.6] · W  = [0.76, 0.22, 0.53]

Row 3: [0.2, -0.3, 0.8] · W = [0.41, -0.06, 0.59]

Result:
H_transformed = [[0.65, 0.22, 0.02],
                 [0.76, 0.22, 0.53],
                 [0.41, -0.06, 0.59]]

Shape: (3, 3)
```

---

### 4.2: Calculate Attention Scores

**Query (decoder hidden state):**
```
q = [0.6, -0.4, 0.5]
Shape: (1, 3)
```

**Expand query for broadcasting:**
```
q_expanded = [[0.6, -0.4, 0.5]]
Shape: (1, 1, 3)
```

**Compute scores (dot product with each encoder output):**
```
score[i] = q · H_transformed[i]

score[0] = [0.6, -0.4, 0.5] · [0.65, 0.22, 0.02]
         = 0.6×0.65 + (-0.4)×0.22 + 0.5×0.02
         = 0.39 - 0.088 + 0.01
         = 0.312

score[1] = [0.6, -0.4, 0.5] · [0.76, 0.22, 0.53]
         = 0.6×0.76 + (-0.4)×0.22 + 0.5×0.53
         = 0.456 - 0.088 + 0.265
         = 0.633

score[2] = [0.6, -0.4, 0.5] · [0.41, -0.06, 0.59]
         = 0.6×0.41 + (-0.4)×(-0.06) + 0.5×0.59
         = 0.246 + 0.024 + 0.295
         = 0.565

Scores = [0.312, 0.633, 0.565]
Shape: (3,)
```

**Interpretation:** 
- Word "love" (score=0.633) gets highest attention
- Word "you" (score=0.565) gets second highest
- Word "I" (score=0.312) gets lowest

---

### 4.3: Apply Softmax to Get Attention Weights

```
Softmax formula: α[i] = exp(score[i]) / Σ(exp(score[j]))

exp(scores) = [exp(0.312), exp(0.633), exp(0.565)]
            = [1.366, 1.883, 1.760]

Sum = 1.366 + 1.883 + 1.760 = 5.009

Attention weights (α):
α = [1.366/5.009, 1.883/5.009, 1.760/5.009]
  = [0.273, 0.376, 0.351]

Shape: (3,)
```

**Interpretation:**
- 27.3% attention on "I"
- **37.6% attention on "love"** ← Highest focus!
- 35.1% attention on "you"

**This makes sense!** When generating "t'aime" (love), the model focuses most on "love" in the input.

---

### 4.4: Calculate Context Vector

**Weighted sum of encoder outputs:**
```
Context vector = Σ(α[i] × H[i])

c = 0.273 × [0.3, 0.7, -0.2] + 
    0.376 × [0.5, 0.4, 0.6] + 
    0.351 × [0.2, -0.3, 0.8]

Calculation:
Dimension 1: 0.273×0.3 + 0.376×0.5 + 0.351×0.2 = 0.082 + 0.188 + 0.070 = 0.340
Dimension 2: 0.273×0.7 + 0.376×0.4 + 0.351×(-0.3) = 0.191 + 0.150 - 0.105 = 0.236
Dimension 3: 0.273×(-0.2) + 0.376×0.6 + 0.351×0.8 = -0.055 + 0.226 + 0.281 = 0.452

Context vector:
c = [0.340, 0.236, 0.452]
Shape: (1, 3)
```

---

## STEP 5: Combine Context with Decoder Output

```
Decoder hidden state: h_dec = [0.6, -0.4, 0.5]
Context vector:       c     = [0.340, 0.236, 0.452]

Concatenate:
combined = [0.6, -0.4, 0.5, 0.340, 0.236, 0.452]
Shape: (1, 6)
```

---

## STEP 6: Final Transformation

### Apply tanh transformation:
```
Weight matrix Wc:
Wc = [[0.5, -0.3, 0.2, 0.1, 0.4, -0.2],
      [0.3, 0.6, -0.1, 0.5, -0.3, 0.2],
      [-0.2, 0.4, 0.7, -0.3, 0.1, 0.5]]

Shape: (3, 6)

output = Wc · combined^T
       = [...calculation...]
       = [0.345, -0.123, 0.567]

Apply tanh:
output_tanh = tanh([0.345, -0.123, 0.567])
            = [0.332, -0.122, 0.513]

Shape: (1, 3)
```

### Final prediction layer:
```
Weight matrix Wo (maps to vocabulary):
Wo = random matrix of shape (vocab_size_fr=1200, 3)

Logits = Wo · output_tanh^T
Shape: (1, 1200)

After softmax:
probabilities = softmax(logits)

Example output (top 3 words):
Word ID 567 ("t'aime"): 0.342  ← Highest probability!
Word ID 234 ("adore"):  0.198
Word ID 891 ("aime"):   0.156
...
```

**Predicted word:** "t'aime"

---

## Summary of the Complete Flow

```
INPUT: "I love you" → Embedded → LSTM Encoder
                                      ↓
                              Encoder Outputs H
                              [3 hidden states]
                                      ↓
DECODER: "Je" → Embedded → LSTM → decoder_hidden (query)
                                      ↓
                        ┌─────────────┴─────────────┐
                        │   ATTENTION MECHANISM      │
                        │                            │
                        │  1. Transform H with W     │
                        │  2. Score = query · H_tr   │
                        │  3. Softmax → weights α    │
                        │     [0.273, 0.376, 0.351]  │
                        │  4. Context = Σ(α × H)     │
                        └─────────────┬─────────────┘
                                      ↓
                        Concat [context, decoder_out]
                                      ↓
                              tanh(Wc · concat)
                                      ↓
                              softmax(Wo · ...)
                                      ↓
                        OUTPUT: "t'aime" (love)
```

---

## Key Characteristics of Luong Attention

1. **"General" scoring function**: Uses learnable weight matrix W
   - `score(h_t, h_s) = h_t^T · W · h_s`

2. **Post-LSTM attention**: Attention is calculated AFTER the decoder LSTM processes the input

3. **Simpler than Bahdanau**: 
   - Bahdanau uses concatenation + tanh + another layer
   - Luong directly uses dot product (faster computation)

4. **Alignment**: Shows which source words the model focuses on when generating each target word

---

## Why Luong Attention is Effective

- Dynamic Focus**: Different attention weights for each decoding step  
- Handles Variable Lengths**: Works with any input/output sequence length  
- Interpretable**: Attention weights visualize what the model "looks at"  
- Gradient Flow**: Provides direct path for gradients from decoder to encoder  

---

## Conclusion

This numerical example demonstrates how Luong Attention mechanism:
- Computes alignment scores between decoder query and encoder outputs
- Uses softmax to create a probability distribution over input words
- Generates a context vector as a weighted sum of encoder outputs
- Combines context with decoder state to make final predictions

The attention weights (0.273, 0.376, 0.351) show that when generating "t'aime" (love), the model correctly focuses most attention (37.6%) on the word "love" in the input sentence "I love you".

This numerical example demonstrates how Luong Attention mechanism:
- Computes alignment scores between decoder query and encoder outputs
- Uses softmax to create a probability distribution over input words
- Generates a context vector as a weighted sum of encoder outputs
- Combines context with decoder state to make final predictions

The attention weights (0.273, 0.376, 0.351) show that when generating "t'aime" (love), the model correctly focuses most attention (37.6%) on the word "love" in the input sentence "I love you".
