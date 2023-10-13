# regime-modeling-with-NLP

This project used natural language processing (NLP) to study, detect, and predict changes in market regimes to improve existing trading strategies. We used text data from the Federal Open Market Committee (FOMC) meeting minutes and price data of the S&P 500. Some methods we used were non-negative matrix factorization, spherical K-means clustering, and naive Bayes classifier.

This project was done for the requirements of the class 46-924 Natural Language Processing. The group members are Dhruv Baid, Prajakta Phadke, and Uday Sharma.

# Abstract

This project used natural language processing (NLP) to study, detect, and predict changes in market regimes to improve existing trading strategies. We used text data from the Federal Open Market Committee (FOMC) meeting minutes and price data of the S\&P 500. We used Non-Negative Matrix Factorization (NMF) to study the characteristics of regimes. We labeled regimes using a hidden Markov model and spherical K-means clustering. Using the labels as the response and price as covariates, we fit a naive Bayes model to predict the regime. We also try adding the NMF model topic weights and word embeddings from FinBERT as covariates. The performance was evaluated on the test set by comparing the metrics of a regime-dependent trading strategy for different combinations of labeling methods and covariates. Using NMF, we found that text data helped identify regimes that corresponded to different macroeconomic topics. However, neither the regimes identified using text data nor the text covariates were able to improve the performance of the trading strategy.

# Research Questions

- Do the regimes, as identified by unsupervised NLP methods, show any distinct characteristics?
- Does using text data in identifying regimes improve the performance of our trading strategy?
- Does using text data as covariates to predict regime help improve our trading strategy?
