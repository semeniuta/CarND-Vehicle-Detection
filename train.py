from sklearn import model_selection, svm, naive_bayes, tree, ensemble, neighbors
import pickle
import datetime

import vdetect


def get_classifier_name(clf):
    return str(clf.__class__)[1:-1].split()[1][1:-1].split('.')[-1]


def save_obj(obj, fname):

    with open(fname, 'wb') as f:
        pickle.dump(clf, f)


if __name__ == '__main__':

    imfiles_0 = vdetect.create_image_files_list('../non-vehicles')
    imfiles_1 = vdetect.create_image_files_list('../vehicles')

    X_train, y_train, X_test, y_test, scaler = vdetect.prepare_train_test_data(imfiles_0[:200], imfiles_1[:200])

    classifiers = [
        #naive_bayes.MultinomialNB(),
        #naive_bayes.GaussianNB(),
        tree.DecisionTreeClassifier(),
        ensemble.AdaBoostClassifier(),
        ensemble.RandomForestClassifier(),
        neighbors.KNeighborsClassifier()
    ]

    clf = tree.DecisionTreeClassifier()

    print('Training ', get_classifier_name(clf))
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    print('Score of {}: {:.3f}'.format(get_classifier_name(clf), score))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    save_obj(clf, 'model_' + timestamp + '.p')
    save_obj(scaler, 'scaler_' + timestamp + '.p')
