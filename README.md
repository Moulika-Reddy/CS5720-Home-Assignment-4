# CS5720-Home-Assignment-4

## Student Information

- **Name:** Moulika Reddy Karra
- **University:** University of Central Missouri
- **Course:** CS5720 — Neural Networks and Deep Learning
- **Semester:** Fall 2026
- **Assignment:** Home Assignment 4

## Assignment Overview

This assignment focuses on understanding sequence-to-sequence models, attention mechanisms, Transformer architecture, and implementing basic Natural Language Processing (NLP) techniques using Python.


## Programming Question 1: NLP Preprocessing Pipeline

**File:** `programming_q1.py`

### Objective

Implement an NLP preprocessing pipeline using the NLTK library.

### Implementation

The program performs three operations:

1. Tokenizes the input sentence into words and punctuation.
2. Removes common English stopwords.
3. Applies Porter stemming to the remaining tokens.

**Input sentence:**

"NLP techniques are used in virtual assistants like Alexa and Siri."

### Results

The program successfully displays the original tokens, tokens after stopword removal, and stemmed words.

The output demonstrates how NLP preprocessing prepares text for further analysis.

## Programming Question 2: Named Entity Recognition

**File:** `programming_q2.py`

### Objective

Use the spaCy library to identify named entities in a sentence.

### Implementation

The program uses the `en_core_web_sm` pretrained English model to identify entities and display:

- Entity text
- Entity label
- Starting character position
- Ending character position

### Results

The model identified entities such as LeBron James (PERSON), NBA (ORG), the United States (GPE), and 2020 (DATE).

The results demonstrate how NER identifies important information from unstructured text. Some predictions may be imperfect because the model uses pretrained statistical patterns.

## Programming Question 3: Scaled Dot-Product Attention

**File:** `programming_q3.py`

### Objective

Implement scaled dot-product attention using NumPy.

### Implementation

The program:

1. Calculates attention scores using the dot product of Q and the transpose of K.
2. Scales the scores by the square root of the key dimension.
3. Applies softmax to calculate attention weights.
4. Multiplies the attention weights by V to calculate the final output.

### Results

**Attention weights:**

[[0.731 0.269]
 [0.269 0.731]]

**Final output:**


[[2.076 3.076 4.076 5.076]
 [3.924 4.924 5.924 6.924]]

The output demonstrates how scaled dot-product attention calculates the importance of different input representations.

## Technologies Used

- Python
- NumPy
- NLTK
- spaCy
- Visual Studio Code
- Conda environment (`cs5720`)

## Installation and Execution

Install the required Python libraries:

```bash
python -m pip install -r requirements.txt
```

Download the spaCy English language model:

```bash
python -m spacy download en_core_web_sm
```

Run each program:

```bash
python programming_q1.py
python programming_q2.py
python programming_q3.py
```

The first program automatically downloads the required NLTK resources if they are not already installed.

## Screenshots

Screenshots of the three programming tasks are included in the `screenshots` folder to show the execution results.

## Conclusion

This assignment helped me understand the concepts of attention mechanisms and Transformers, along with practical NLP techniques such as tokenization, stopword removal, stemming, and named entity recognition. Implementing scaled dot-product attention also helped me understand how attention weights and output representations are calculated.