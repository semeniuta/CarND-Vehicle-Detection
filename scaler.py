from sklearn.preprocessing import StandardScaler


class FeatureScaler(object):

    def __init__(self, training_sets_dict):
        self._scalers = {k: StandardScaler().fit(x) for k, x in training_sets_dict.items()}

    def scale(self, vecs_dict):

        transformed_vecs = dict()

        for k in vecs_dict.keys():

            x = vecs_dict[k]
            scaler = self._scalers[k]

            xt = scaler.transform(x)
            transformed_vecs[k] = xt

        return transformed_vecs
