from sklearn.neighbors import KNeighborsClassifier

def knn(n_neighbors, train_matrix, train_categorie, test_matrix, test_categorie):
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(train_matrix, train_categorie)
    predict_categorie = knn.predict(test_matrix)
    
    num_sum = len(test_categorie)
    num_correct = sum(predict_categorie == test_categorie)
    num_wrong = num_sum - num_correct

    print('KNN : %f, with %d correct and %d wrong.(neighbor = %d)' %(float(num_correct)/(num_sum), num_correct, num_wrong, n_neighbors))

