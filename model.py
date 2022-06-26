from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import check_random_state
from utils import *
from scipy.optimize import minimize, OptimizeResult, differential_evolution
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.losses import BinaryCrossentropy


class DenseSequential(Sequential):
    def __init__(self, input_dim, output_dim, num_layers, num_units, layer_kws={}, final_layer_kws={}):
        super(DenseSequential, self).__init__()

        for i in range(num_layers):
            if not i:
                self.add(Dense(num_units, input_dim=input_dim, **layer_kws))
            self.add(Dense(num_units, **layer_kws))

        self.add(Dense(output_dim, **final_layer_kws))


class MaximizableMixin:
    def __init__(self, transform=tf.identity, *args, **kwargs):
        super(MaximizableMixin, self).__init__(*args, **kwargs)
        # negate to turn into minimization problem for ``scipy.optimize``
        self._func_min = convert(self, transform=lambda u: transform(-u))

    def maxima(self, bounds, filter_fn=lambda res: True, num_samples=1024, random_state=None, nasbench=False):
        random_state = check_random_state(random_state)

        (low, high), dim = from_bounds(bounds)

        if not nasbench:
            X_init = random_state.uniform(low=low, high=high, size=(num_samples, dim))
        else:
            # Sample one-hot initializations for nasbenc201
            X_init = []
            for _ in range(num_samples):
                x = np.zeros((6, 5))
                idx = random_state.choice(5, size=6)
                for i in range(6):
                    x[i][idx[i]] = 1
                X_init.append(x.reshape(-1))
            X_init = np.array(X_init)

        z_init = self.predict(X_init).squeeze(axis=-1)
        # the function to minimize is negative of the classifier output
        f_init = - z_init

        i = np.argmin(f_init, axis=None)
        result = OptimizeResult(x=X_init[i], fun=f_init[i], success=True)
        if filter_fn(result):
            return result
        # Otherwise, we may try to return the second best choice or use a random choice for exploration


class MaximizableDenseSequential(MaximizableMixin, DenseSequential):
    pass


class MaximizableXGBClassifier(XGBClassifier):

    def maxima(self, bounds, filter_fn=lambda res: True, num_samples=1024, random_state=None, nasbench=False):

        self._func_min = lambda u: -self.predict_proba(u)[:, 1]
        random_state = check_random_state(random_state)
        (low, high), dim = from_bounds(bounds)

        if not nasbench:
            X_init = random_state.uniform(low=low, high=high, size=(num_samples, dim))
        else:
            # Sample one-hot initializations for nasbenc201
            X_init = []
            for _ in range(num_samples):
                x = np.zeros((6, 5))
                idx = random_state.choice(5, size=6)
                for i in range(6):
                    x[i][idx[i]] = 1
                X_init.append(x.reshape(-1))
            X_init = np.array(X_init)

        z_init = self.predict_proba(X_init)[:, 1]
        # the function to minimize is negative of the classifier output
        f_init = - z_init

        i = np.argmin(f_init, axis=None)
        result = OptimizeResult(x=X_init[i], fun=f_init[i], success=True)
        if filter_fn(result):
            return result


class MaximizableRFClassifier(RandomForestClassifier):
    def maxima(self, bounds, filter_fn=lambda res: True, num_samples=1024, random_state=None, nasbench=False):
        self._func_min = lambda u: -self.predict_proba(u)[:, 1]
        random_state = check_random_state(random_state)
        (low, high), dim = from_bounds(bounds)

        if not nasbench:
            X_init = random_state.uniform(low=low, high=high, size=(num_samples, dim))
        else:
            # Sample one-hot initializations for nasbenc201
            X_init = []
            for _ in range(num_samples):
                x = np.zeros((6, 5))
                idx = random_state.choice(5, size=6)
                for i in range(6):
                    x[i][idx[i]] = 1
                X_init.append(x.reshape(-1))
            X_init = np.array(X_init)

        z_init = self.predict_proba(X_init)[:, 1]
        # the function to minimize is negative of the classifier output
        f_init = - z_init

        i = np.argmin(f_init, axis=None)
        result = OptimizeResult(x=X_init[i], fun=f_init[i], success=True)
        if filter_fn(result):
            return result


class Record:
    def __init__(self):
        self.features = []
        self.targets = []
        self.budgets = []

    def size(self):
        return len(self.targets)

    def append(self, x, y, b=None):
        self.features.append(x)
        self.targets.append(y)
        if b is not None:
            self.budgets.append(b)

    def load_classification_data(self, gamma, weight_type):
        assert weight_type in ["pi", "ei"]

        if weight_type == "ei":
            X, y = np.vstack(self.features), np.hstack(self.targets)
            tau = np.quantile(y, q=gamma)
            z = np.less(y, tau)
            x1, z1 = X[z], z[z]
            x0, z0 = X, np.zeros_like(z)
            w1 = (tau - y)[z]
            w1 = w1 / np.mean(w1)
            w0 = 1 - z0

            x = np.concatenate([x1, x0], axis=0)
            z = np.concatenate([z1, z0], axis=0)
            s1 = x1.shape[0]
            s0 = x0.shape[0]

            w = np.concatenate([w1 * (s1 + s0) / s1, w0 * (s1 + s0) / s0], axis=0)
            w = w / np.mean(w)
            return x, z, w

        if weight_type == "pi":
            x, y = np.vstack(self.features), np.hstack(self.targets)
            tau = np.quantile(y, q=gamma)
            z = np.less(y, tau)
            return x, z, np.ones_like(z)

    def is_duplicate(self, x, rtol=1e-5, atol=1e-8):
        return any(np.allclose(x_prev, x, rtol=rtol, atol=atol) for x_prev in self.features)


class LFBO:
    def __init__(self, config_space, num_random_init, gamma, weight_type, model_type, nasbench,
                 num_samples=5000, method="L-BFGS-B", seed=0):
        self.config_space = DenseConfigurationSpace(config_space, seed=seed)
        self.num_random_init = num_random_init
        self.nasbench = nasbench
        self.model_type = model_type
        self.weight_type = weight_type
        self.gamma = gamma
        self.record = Record()
        self.bounds = self.config_space.get_bounds()
        self.num_samples = num_samples
        self.method = method
        self.random_state = np.random.RandomState(seed)
        self.model = self.construct_model()

    def construct_model(self):
        if self.model_type == 'mlp':
            model = MaximizableDenseSequential(input_dim=self.config_space.get_dimensions(sparse=False),
                                               output_dim=1, num_layers=2, num_units=32,
                                               layer_kws=dict(activation="relu",
                                                              kernel_regularizer=None,
                                                              bias_regularizer=None))
            model.compile(optimizer="adam", metrics=["accuracy"],
                          loss=BinaryCrossentropy(from_logits=True))
            model.summary(print_fn=print)
            return model
        elif self.model_type == 'rf':
            model = MaximizableRFClassifier(n_estimators=1000, min_samples_split=2)
            return model
        elif self.model_type == 'xgb':
            model = MaximizableXGBClassifier(objective='binary:logistic', min_child_weight=1,
                                             learning_rate=0.3, n_estimators=100)
            return model
        else:
            raise NotImplementedError

    def _is_unique(self, res):
        is_duplicate = self.record.is_duplicate(res.x)
        return not is_duplicate

    def step(self):
        config_random = self.config_space.sample_configuration()
        config_random_dict = config_random.get_dictionary()

        if self.record.size() < self.num_random_init:
            return config_random_dict

        # Update the classifier
        x, z, w = self.record.load_classification_data(self.gamma, self.weight_type)
        if self.model_type == 'mlp':
            dataset_size = self.record.size()
            batch_size = 64
            num_steps = steps_per_epoch(dataset_size, batch_size)
            num_steps_per_iter = 100
            num_epochs_per_iter = num_steps_per_iter // num_steps
            self.model.fit(x, z, sample_weight=w, epochs=num_epochs_per_iter, batch_size=batch_size, callbacks=[], verbose=False)
        elif self.model_type == 'rf':
            self.model.fit(x, z, sample_weight=w)

        elif self.model_type == 'xgb':
            self.model.fit(x, z, sample_weight=w, eval_metric='logloss', callbacks=[], verbose=False)
        else:
            raise NotImplementedError

        opt = self.model.maxima(bounds=self.bounds,
                                filter_fn=self._is_unique,
                                num_samples=self.num_samples,
                                random_state=self.random_state,
                                nasbench=self.nasbench)

        if opt is None:
            return config_random_dict

        loc = opt.x
        config_opt_arr = maybe_distort(loc, None, self.bounds, self.random_state)
        config_opt_dict = dict_from_array(self.config_space, config_opt_arr)

        return config_opt_dict

    def add_new_observation(self, config_dict, y):
        config_arr = array_from_dict(self.config_space, config_dict)
        self.record.append(x=config_arr, y=y)
