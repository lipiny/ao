from sklearn import svm

def svmclf(train_matrix, train_categorie, test_matrix, test_categorie):
    clf = svm.SVC()
    clf.fit(train_matrix, train_categorie)
    predict_categorie = clf.predict(test_matrix)
    
    num_sum = len(test_categorie)
    num_correct = sum(predict_categorie == test_categorie)
    num_wrong = num_sum - num_correct

    print('SVM : %f, with %d correct and %d wrong.' %(float(num_correct)/(num_sum), num_correct, num_wrong))
