from sklearn import model_selection, svm, naive_bayes, tree, ensemble, neighbors
import pickle

import vdetect


def get_classifier_name(clf):
    return str(clf.__class__)[1:-1].split()[1][1:-1].split('.')[-1]


def save_model(clf, fname):

    with open('first.p', 'wb') as f:
        pickle.dump(clf, fname)


if __name__ == '__main__':

    imfiles_0 = vdetect.create_image_files_list('../non-vehicles')
    imfiles_1 = vdetect.create_image_files_list('../vehicles')

    X_train, y_train, X_test, y_test = vdetect.prepare_train_test_data(imfiles_0, imfiles_1)

    classifiers = [
        #naive_bayes.MultinomialNB(),
        #naive_bayes.GaussianNB(),
        tree.DecisionTreeClassifier(),
        ensemble.AdaBoostClassifier(),
        ensemble.RandomForestClassifier(),
        neighbors.KNeighborsClassifier()
    ]

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    print(get_classifier_name(clf), score)
