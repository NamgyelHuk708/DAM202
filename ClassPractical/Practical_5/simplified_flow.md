# Luong Attention: Simplified Flow with Key Calculations

## COMPLETE FLOW DIAGRAM WITH NUMERICAL EXAMPLE

### Input: "I love you" → Output: "Je t'aime"

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENGLISH INPUT: "I love you"                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   WORD EMBEDDING       │
                    │   [45, 123, 78]        │
                    │         ↓              │
                    │   [[0.5, 0.2, -0.3, 0.1],  │
                    │    [0.8, -0.1, 0.4, 0.2],  │
                    │    [-0.2, 0.6, 0.3, -0.5]] │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  ENCODER (Bi-LSTM)     │
                    │  Processes sequence    │
                    └────────────┬───────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────┐
              │    ENCODER OUTPUTS (H)           │
              │                                  │
              │  H = [[0.3, 0.7, -0.2],   ← "I"     │
              │       [0.5, 0.4, 0.6],    ← "love"  │
              │       [0.2, -0.3, 0.8]]   ← "you"   │
              │                                  │
              │  Last state: h=[0.2,-0.3,0.8]    │
              └──────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │    DECODER INPUT: "Je"           │
              │    Embedded: [0.4, -0.2, 0.3, 0.1]│
              └──────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │    DECODER LSTM                  │
              │    Output (query):               │
              │    q = [0.6, -0.4, 0.5]         │
              └──────────────┬───────────────────┘
                             │
                             ▼
╔═════════════════════════════════════════════════════════════════════════╗
║                     ATTENTION MECHANISM                                ║
╚═════════════════════════════════════════════════════════════════════════╝
                             │
                             ▼
              ┌──────────────────────────────────┐
              │  STEP 1: Transform Encoder       │
              │                                  │
              │  H_transformed = H · W           │
              │                                  │
              │  [[0.65, 0.22, 0.02],           │
              │   [0.76, 0.22, 0.53],           │
              │   [0.41, -0.06, 0.59]]          │
              └──────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │  STEP 2: Calculate Scores        │
              │  (Dot product query with each H) │
              │                                  │
              │  score[0] = [0.6,-0.4,0.5]·[0.65,0.22,0.02] │
              │           = 0.312                │
              │                                  │
              │  score[1] = [0.6,-0.4,0.5]·[0.76,0.22,0.53] │
              │           = 0.633  ← HIGHEST!    │
              │                                  │
              │  score[2] = [0.6,-0.4,0.5]·[0.41,-0.06,0.59]│
              │           = 0.565                │
              │                                  │
              │  Scores = [0.312, 0.633, 0.565]  │
              └──────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │  STEP 3: Apply Softmax           │
              │                                  │
              │  exp(scores) = [1.366, 1.883, 1.760] │
              │  sum = 5.009                     │
              │                                  │
              │  α = [1.366/5.009,              │
              │       1.883/5.009,              │
              │       1.760/5.009]              │
              │                                  │
              │  α = [0.273, 0.376, 0.351]      │
              │       ↑       ↑       ↑         │
              │      27.3%  37.6%   35.1%       │
              │       "I"   "love"  "you"       │
              └──────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │  STEP 4: Context Vector          │
              │  (Weighted sum)                  │
              │                                  │
              │  c = 0.273×[0.3,0.7,-0.2] +     │
              │      0.376×[0.5,0.4,0.6] +      │
              │      0.351×[0.2,-0.3,0.8]       │
              │                                  │
              │  c = [0.340, 0.236, 0.452]      │
              │                                  │
              │  (Most weighted by "love"!)      │
              └──────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │  STEP 5: Concatenate             │
              │                                  │
              │  [decoder_output | context]      │
              │  [0.6,-0.4,0.5 | 0.340,0.236,0.452]│
              │                                  │
              │  combined = [0.6,-0.4,0.5,0.340,0.236,0.452]│
              └──────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │  STEP 6: Final Dense Layers      │
              │                                  │
              │  output = tanh(Wc · combined)    │
              │         = [0.332, -0.122, 0.513] │
              │                                  │
              │  logits = Wo · output            │
              │  probs = softmax(logits)         │
              └──────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │     PREDICTED WORD               │
              │                                  │
              │  Word ID 567: "t'aime" (34.2%)  │
              │  Word ID 234: "adore"  (19.8%)  │
              │  Word ID 891: "aime"   (15.6%)  │
              │                                  │
              │  OUTPUT: "t'aime"                │
              └──────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │   FINAL TRANSLATION              │
              │   "Je t'aime"                    │
              └──────────────────────────────────┘
```

---

##  KEY CALCULATIONS SUMMARY

### 1. **Score Calculation (Dot Product)**
```
For "I":    0.6×0.65 + (-0.4)×0.22 + 0.5×0.02 = 0.312
For "love": 0.6×0.76 + (-0.4)×0.22 + 0.5×0.53 = 0.633 ← HIGHEST!
For "you":  0.6×0.41 + (-0.4)×(-0.06) + 0.5×0.59 = 0.565
```

### 2. **Softmax (Attention Weights)**
```
exp([0.312, 0.633, 0.565]) = [1.366, 1.883, 1.760]
Sum = 5.009

Attention weights α:
- "I":    1.366 / 5.009 = 0.273 (27.3%)
- "love": 1.883 / 5.009 = 0.376 (37.6%) ← MAXIMUM ATTENTION!
- "you":  1.760 / 5.009 = 0.351 (35.1%)
```

### 3. **Context Vector (Weighted Sum)**
```
c = 0.273 × [0.3, 0.7, -0.2]    (contribution from "I")
  + 0.376 × [0.5, 0.4, 0.6]     (contribution from "love" - largest!)
  + 0.351 × [0.2, -0.3, 0.8]    (contribution from "you")
  
= [0.340, 0.236, 0.452]
```

---

## KEY INSIGHT

**When generating "t'aime" (love), the attention mechanism:**

- Correctly focuses **37.6%** attention on "love" in the input  
- Distributes remaining attention to contextual words ("you" and "I")  
- Creates a context vector weighted most heavily by the relevant word  

This demonstrates how attention allows the decoder to **dynamically focus** on the most relevant parts of the input sequence!

---

## Mathematical Formula Summary

**Luong Attention (General)**:
```
1. Transform:  H' = H · W
2. Score:      s = query^T · H'
3. Weights:    α = softmax(s)
4. Context:    c = Σ(α[i] × H[i])
5. Output:     o = tanh(Wc · [c; decoder_output])
6. Prediction: y = softmax(Wo · o)
```

Where:
- `H`: Encoder outputs (3×3 matrix)
- `W`: Learnable transformation matrix (3×3)
- `query`: Decoder hidden state (1×3)
- `α`: Attention weights (1×3)
- `c`: Context vector (1×3)
