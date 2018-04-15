from sklearn import model_selection, svm, naive_bayes, tree, ensemble, neighbors
import pickle
import datetime
import os
import json

import vdetect


def get_classifier_name(clf):
    return str(clf.__class__)[1:-1].split()[1][1:-1].split('.')[-1]


def save_obj(obj, fname):

    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def save_json(data, fname):

    with open(fname, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':

    imfiles_0 = vdetect.create_image_files_list('../non-vehicles')
    imfiles_1 = vdetect.create_image_files_list('../vehicles')

    print('Number of vehicle examples:', len(imfiles_1))
    print('Number of non-vehicle examples:', len(imfiles_0))

    hyperparams = {
        'hog_n_orient': 9,
        'hog_cell_sz': 8,
        'hog_block_sz': 4,
        'binning_sz': 8,
        'hist_bins': 64
    }

    X_train, y_train, X_test, y_test, scaler, fs_sizes = vdetect.prepare_train_test_data(
        imfiles_0,
        imfiles_1,
        hyperparams
    )

    print('Dimensionality of feature sets:', fs_sizes)
    print('Training data shape:', X_train.shape, y_train.shape)
    print('Testing data shape:', X_test.shape, y_test.shape)

    classifiers = {
        'decision_tree_default': tree.DecisionTreeClassifier(),
        'decision_tree_bigger_split': tree.DecisionTreeClassifier(
            min_samples_split=5
        ),
        #'svc_custom': svm.SVC(
        #    C=1.,
        #    gamma=0.1
        #),
        #'naive_bayes': naive_bayes.GaussianNB(),
        'random_forest_default': ensemble.RandomForestClassifier(),
        'grad_boost_default': ensemble.GradientBoostingClassifier(),
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

    clf_fname = os.path.join(savedir, 'classifiers.p')
    save_obj(classifiers, clf_fname)
    print('Saved classifier to {}'.format(clf_fname))

    scaler_fname = os.path.join(savedir, 'scaler.p')
    save_obj(scaler, scaler_fname)
    print('Saved scaler to {}'.format(scaler_fname))

    hp_fname = os.path.join(savedir, 'hyper.json')
    save_obj(hyperparams, hp_fname)
    print('Saved hyperparams to {}'.format(hp_fname))
