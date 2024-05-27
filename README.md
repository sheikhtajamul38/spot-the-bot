# Spot the Bot

## Overview
**Spot the Bot** is a project focused on detecting bot-generated texts by analyzing semantic trajectories of natural language texts. This research primarily investigates the Kanuri and Miskito languages using advanced word embedding techniques and clustering algorithms. The goal is to develop effective mechanisms to distinguish between human-authored and bot-generated content, addressing the rising concerns of misinformation in digital communications.

## Features
- **Data Collection and Pre-processing**: Collection of text corpora for Kanuri and Miskito languages, followed by extensive cleaning and preprocessing.
- **Text Generation**: Generation of bot-like texts using ChatGPT 3.5 to simulate bot-generated content.
- **Word Embeddings**: Implementation of Word2Vec and FastText models to create meaningful vector representations of words.
- **Clustering**: Application of the Wishart clustering algorithm to identify distinct clusters of human and bot-generated texts.
- **Statistical Analysis**: Use of Mann-Whitney statistical tests and Holm-Bonferroni corrections to validate the significance of clustering results.

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Libraries: `numpy`, `pandas`, `scikit-learn`, `gensim`, `scipy`, `matplotlib`, `nltk`
- Access to ChatGPT 3.5 for text generation

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/username/spot-the-bot.git
