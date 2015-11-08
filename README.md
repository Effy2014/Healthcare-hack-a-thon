# Healthcare-hack-a-thon
Modeling Team 1: Predicting Likelihood on Media Coverage of Healthcare Article
Data Given
Reuters & Jama press release of healthcare articles, as well as the ones not covered by media
 
Initial Challenge
Non of the raw data is numerical; 
Matches between covered & non-covered. How to use this similarity information
 
Processing
Using number of matches as indicator of the competition. Intuitive: if similar topics have been released multiple times, maybe it's not that newsworthy.
Calculating TF-IDF matrix on the keywords from abstract; MeSH terms and citation titles. Performed PCA to reduce the number of new features
Dummy variable for authors
 
Models in Mind
SVM/ LDA/ Regularized Lasso Regression/ Naive Bayes
 
Challenges
Similarities aren't transitive-- A close to B & B close to C doesn't necessarily mean A close to C
PCA too slow (3000 feature vectors)
Number of unique authors > number of articles. Unexpected
 
Alternative Approach
Discard the matches column. Interaction between 0 and 1 is interesting information but could be already covered from the existing columns, such as cosine similarity/Jaccard coefficient
Naive Bayes, using True/False, rather than TF-IDF score.
Logistic Regression with Lasso
Number of authors for each article as new feature
 
Result
Initial Logistic Regression got 98.28% correction, when using TF-IDF features only from Abstract column
 
Takeaways
When you have limited time & exposure to a dataset, bottom-up from simplicity. 
Quick summaries of raw data is usually better than assumptions about a new field
Be flexible when dealing with real world data. This is especially true when you are also high with adrenaline
And yes it is enormously fun

