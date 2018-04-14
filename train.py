from sklearn import model_selection, svm, naive_bayes, tree, ensemble, neighbors
import pickle
import datetime
import os

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

    classifiers = {
        'decision_tree_default': tree.DecisionTreeClassifier(),
        'decision_tree_bigger_split': tree.DecisionTreeClassifier(
            min_samples_split=10
        ),
        'svc_custom': svm.SVC(
            C=0.25
        ),
        'random_forest_default': ensemble.RandomForestClassifier(),
    }


    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    for k, clf in classifiers.items():

        print('Training {} ({})'.format(k, get_classifier_name(clf)))
        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)

        print('Score of {}: {:.3f}'.format(k, score))


    savedir = 'serialize/{}'.format(timestamp)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    clf_fname = os.path.join(savedir, 'classifiers_{}.p'.format(timestamp))
    save_obj(classifiers, clf_fname)
    print('Saved classifier to {}'.format(clf_fname))

    scaler_fname = os.path.join(savedir, 'scaler_{}.p'.format(timestamp))
    save_obj(scaler, scaler_fname)
    print('Saved scaler to {}'.format(scaler_fname))
