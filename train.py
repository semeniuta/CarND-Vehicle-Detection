from sklearn import model_selection, svm, naive_bayes, tree, ensemble, neighbors
import pickle
import datetime

import vdetect


def get_classifier_name(clf):
    return str(clf.__class__)[1:-1].split()[1][1:-1].split('.')[-1]


def save_obj(obj, fname):

    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':

    imfiles_0 = vdetect.create_image_files_list('../non-vehicles')
    imfiles_1 = vdetect.create_image_files_list('../vehicles')

    print('Number of vehicle examples:', len(imfiles_1))
    print('Number of non-vehicle examples:', len(imfiles_0))

    X_train, y_train, X_test, y_test, scaler = vdetect.prepare_train_test_data(imfiles_0, imfiles_1)

    print('Training data shape:', X_train.shape, y_train.shape)
    print('Testing data shape:', X_test.shape, y_test.shape)

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
    save_obj(clf, 'serialize/model_' + timestamp + '.p')
    save_obj(scaler, 'serialize/scaler_' + timestamp + '.p')
