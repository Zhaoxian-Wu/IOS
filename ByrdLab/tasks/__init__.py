class Task:
    def __init__(self, weight_decay, data_package,
                 model, loss_fn, test_fn,
                 get_train_iter=None, get_test_iter=None,
                 initialize_fn=None, super_params={},
                 name='', model_name=''):
        if model_name == '':
            model_name = model._get_name()
        if name == '':
            name = model_name + '_' + data_package.name
        self.weight_decay = weight_decay
        self.data_package = data_package
        self.model = model
        self.loss_fn = loss_fn
        self.test_fn = test_fn
        self.initialize_fn = initialize_fn
        self.get_train_iter = get_train_iter
        self.get_test_iter = get_test_iter
        self.super_params = super_params
        self.name = name
        self.model_name = model_name