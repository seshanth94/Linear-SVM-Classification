1. The required libraries are imported. Then the data to be trained & tested is imported using pandas.
2. We are required to predict position of the player and thus this is the class label.
3. The essential features are taken into a set to improve the accuracy.
4. The data is split according to 72/25 where 75% will be used for building the model and the rest 25% will be used for testing its classification ability. This model is built using Linear SVC Classifier (Using 26 levels, because there are 26 columns of data).
5. The confusion matrix of the above model is printed.
6. A different model is also built using the same Linear SVC Classifier. The only thing different is the way we use the data set to train. Here, 10 Fold Stratifier is used so that model isn’t skewed in accordance to the data.
7. We print the accuracy in each fold and then average them out.
