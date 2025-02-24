# Reiner Luminto AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation System
# Spring 2025

## Overview
This project implements a content-based recommendation system for movies. It uses a TF-IDF vectorizer and cosine similarity to match a user's text query against movie descriptions and reviews, returning the top recommendations.

## Dataset
- **Source:** https://www.kaggle.com/datasets/amanbarthwal/imdb-movies-data (movies.csv consists of only about the first thousand rows of the dataset)
- **Description:** It contains movie details such as Title, Description, Review, Genre, and Rating.

## Setup

### 1. Clone the Repository

```sh
git clone https://github.com/rluminto/lumaa-spring-2025-ai-ml.git
```

### 2. Set Up a Virtual Environment

#### On macOS/Linux:

```sh
python -m venv venv
source venv/bin/activate
```

#### On Windows:

```sh
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
venv\Scripts\activate
```

### 3. Install Dependencies

Once the virtual environment is activated, install the required dependencies:

```sh
pip install -r requirements.txt
```

## Running the Recommendation System

To run the script with a query, use:

```sh
python recommend.py "your-query"
```
Example query: "I love action movies with great speacial effects"

This will output the **top 5 recommended movies** based on text similarity.


## Salary Expectation

**Expected Monthly Salary:** **\$3000 - \$4000**

## Future Improvements

- Enhance recommendation accuracy by incorporating **word embeddings** (e.g., Word2Vec, BERT).
- Add a **user interface** (web or CLI) for interactive queries.
- Expand dataset to include more movies for better generalization.
- Integrate **collaborative filtering** for a hybrid recommendation approach.



