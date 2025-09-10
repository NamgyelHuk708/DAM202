# Practical 2 – Word2Vec Training and Evaluation

This practical demonstrates the complete implementation of Word2Vec models, focusing on:

- **Training custom Word2Vec models** using different architectures (CBOW vs Skip-gram).
- **Understanding neural language model fundamentals** through hands-on implementation.
- **Evaluating model quality** using comprehensive assessment methods.
- **Comparing different training approaches** and their effectiveness.

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Dataset](#2-dataset)
- [3. Preprocessing Pipeline](#3-preprocessing-pipeline)
- [4. Model Training](#4-model-training)
- [5. Evaluation](#5-evaluation)
- [6. Comparison and Results](#6-comparison-and-results)
- [7. Code Structure](#7-code-structure)
- [8. References](#8-references)

---

## 1. Introduction

### 1.1 Practical Objectives

This practical demonstrates the complete implementation of Word2Vec models, focusing on:

- **Training custom Word2Vec models** using different architectures (CBOW vs Skip-gram)
- **Understanding neural language model fundamentals** through hands-on implementation
- **Evaluating model quality** using comprehensive assessment methods
- **Comparing different training approaches** and their effectiveness

### 1.2 Why Enhanced Datasets?

To improve vocabulary coverage and model accuracy, this project uses multiple datasets beyond the original `text.txt`. This approach provides:

- **Richer vocabulary coverage** from diverse text sources.
- **Better semantic relationships** through varied linguistic contexts.
- **Improved model robustness** with a larger training corpus.
- **More reliable evaluation metrics** with comprehensive data.

### 1.3 CBOW vs Skip-gram Comparison

| Architecture | Input | Output | Best For |
| :---: | :---: | :---: | :---: |
| **CBOW** | Context words | Center word | Frequent words, syntactic relationships |
| **Skip-gram** | Center word | Context words | Rare words, semantic relationships |

---

## 2. Dataset

### 2.1 Original Dataset

- **Source:** `text.txt` (Alice's Adventures in Wonderland)
- **Content:** Lewis Carroll's classic novel text.
- **Characteristics:** Literary vocabulary, narrative structure.

### 2.2 Additional Datasets

#### 2.2.1 NLTK Gutenberg Corpus
Classic literature from the NLTK Gutenberg corpus was added to enrich the vocabulary and grammatical variety.

```python
# Enhanced corpus with classic literature
from nltk.corpus import gutenberg
gutenberg_texts = []
for fileid in gutenberg.fileids()[:5]:  # First 5 books
    text = ' '.join(gutenberg.words(fileid))
    gutenberg_texts.extend([text])
```

#### 2.2.2 Text8 Wikipedia Corpus
The `text8` corpus, a ~100MB collection of pre-cleaned Wikipedia text, was used as the primary training data for its scale and broad vocabulary.

```python
# Large-scale Wikipedia corpus
import gensim.downloader as api
corpus = api.load('text8')  # ~100MB of Wikipedia text
```

### 2.3 Dataset Statistics

The combination of `text.txt` and the Gutenberg corpus resulted in the following pre-processing statistics:

- **Total Documents:** 3,603
- **Processed Sentences:** 46,268
- **Vocabulary Size:** 17,268 unique words
- **Average Sentence Length:** 11.6 words
- **Sentence Length Range:** 3-193 words

---

## 3. Preprocessing Pipeline

### 3.1 Advanced Text Preprocessing

A sophisticated `AdvancedTextPreprocessor` class was implemented for comprehensive text cleaning.

```python
class AdvancedTextPreprocessor:
    """Comprehensive text preprocessing for Word2Vec training"""
    
    def __init__(self,
                 lowercase=True,
                 remove_punctuation=True,
                 remove_numbers=True,
                 remove_stopwords=True,
                 lemmatize=True,
                 min_word_length=3,
                 max_word_length=30,
                 remove_urls=True,
                 remove_emails=True,
                 keep_sentences=True):
```

### 3.2 Preprocessing Steps

1.  **Text Cleaning:** URL and email removal, whitespace normalization.
2.  **Tokenization:** Sentence and word-level tokenization using NLTK.
3.  **Filtering & Normalization:** Lowercasing, stopword removal, word length constraints, and lemmatization.
4.  **Quality Control:** Filtering sentences by minimum length.

---

## 4. Model Training

### 4.1 Parameter Selection

#### 4.1.1 Intelligent Parameter Recommendation
A function was implemented to recommend optimal Word2Vec parameters based on corpus characteristics.

```python
def recommend_parameters(corpus_size, vocab_size, domain_type, computing_resources):
    """Recommend optimal Word2Vec parameters based on corpus characteristics"""
```

#### 4.1.2 Training Configuration Used
The models were trained on the `text8` corpus with the following parameters:

- **Vector Size:** 200 dimensions
- **Window Size:** 10 words
- **Min Count:** 10 occurrences
- **Negative Sampling:** 5 samples
- **Epochs:** 3 iterations
- **Workers:** Based on available CPU cores

### 4.2 Training Process

A custom `EpochLogger` callback was used to monitor training progress. Two models were trained: one using the Skip-gram architecture and one using CBOW.

#### 4.2.1 Training Results

**Skip-gram Model Training:**
- **Training Time:** ~16.6 minutes
- **Final Vocabulary:** 47,134 words
- **Model File:** `my_word2vec_model_text8.model`

**CBOW Model Training:**
- **Training Time:** ~3 minutes (approx. 5.5x faster than Skip-gram)
- **Final Vocabulary:** 47,134 words
- **Model File:** `my_word2vec_model_text8_cbow.model`

---

## 5. Evaluation

A `Word2VecEvaluator` class was implemented to assess model quality.

```python
class Word2VecEvaluator:
    """Comprehensive evaluation suite for Word2Vec models"""
    
    def evaluate_word_similarity(self, word_pairs_with_scores):
        """Spearman correlation with human judgments"""
        
    def evaluate_analogies(self, analogy_dataset):
        """Word analogy task accuracy"""
```

### 5.1 Word Similarity Evaluation
The models were evaluated on a word similarity task using Spearman correlation.

- **Skip-gram Spearman Correlation:** 0.6000
- **CBOW Spearman Correlation:** 0.5429

### 5.2 Analogy Tasks
The models were tested on their ability to solve word analogies (e.g., "man is to woman as king is to ?").

- **Skip-gram Accuracy:** 12.5% (1 out of 8 correct)
- **CBOW Accuracy:** 0.0% (0 out of 8 correct)

### 5.3 Qualitative Analysis

**Similarity for "king":**
- **Skip-gram:** ['prince', 'queen', 'throne', 'kingdom', 'ii']
- **CBOW:** ['prince', 'queen', 'throne', 'kingdom', 'reign']

**Similarity for "woman":**
- **Skip-gram:** ['girl', 'man', 'mother', 'she', 'child']
- **CBOW:** ['man', 'girl', 'mother', 'child', 'teenage']

**Analysis:** Both models capture relevant semantic relationships. The Skip-gram model appears slightly better at capturing nuanced relationships in the analogy task, while both are effective at finding similar words.

---

## 6. Comparison and Results

### 6.1 Performance Comparison

| Metric | Skip-gram (sg=1) | CBOW (sg=0) | Winner |
| :---: | :---: | :---: | :---: |
| **Training Time** | ~16.6 min | ~3 min | **CBOW** |
| **Word Similarity (Spearman)** | 0.6000 | 0.5429 | **Skip-gram** |
| **Analogy Accuracy** | 12.5% | 0.0% | **Skip-gram** |
| **Qualitative Similarity** | Good | Good | Tie |

### 6.2 Key Findings

- **CBOW is significantly faster** to train, making it suitable for large datasets where speed is a priority.
- **Skip-gram performs better on semantic tasks** like word similarity and analogy, which aligns with its design to predict context from a word.
- The choice between CBOW and Skip-gram involves a **trade-off between training speed and semantic accuracy**.

---

## 7. Code Structure

```
practical_2/
│
├── Practical2(DAM202)(1).ipynb    # Main implementation notebook
├── README.md                      # This comprehensive report
├── text.txt                       # Original training dataset
│
└── (Generated Models)
    ├── my_word2vec_model_text8.model       # Skip-gram model
    └── my_word2vec_model_text8_cbow.model  # CBOW model
```

---

## 8. References

1.  Mikolov, T. et al. (2013). "Efficient Estimation of Word Representations in Vector Space." *arXiv:1301.3781*
2.  Mikolov, T. et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality." *NIPS 2013*
3.  Goldberg, Y. & Levy, O. (2014). "word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method." *arXiv:1402.3722*
4.  Gensim Documentation: [Word2Vec Tutorial](https://radimrehurek.com/gensim/)
5.  NLTK Documentation: [Natural Language Toolkit](https://www.nltk.org/)
