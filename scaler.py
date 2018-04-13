import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureScaler(object):

    def __init__(self, training_sets_dict):
        self._scalers = {k: StandardScaler().fit(x) for k, x in training_sets_dict.items()}

    def scale(self, vecs_dict):
        '''
        vecs_dict -- dictionary of feature groups ('hog', 'binned', 'hist'),
        preprared by a call to vdetect.extract_features (single image case),
        or by a call to prepare_train_test_data_single_class
        (data preparation for training)
        '''

        transformed_vecs = dict()

        for k in vecs_dict.keys():

            x = vecs_dict[k]
            if len(x.shape) == 1:
                x = np.array(x.reshape(1, -1), dtype=np.float64)

            scaler = self._scalers[k]

            xt = scaler.transform(x)
            transformed_vecs[k] = xt

        return transformed_vecs
