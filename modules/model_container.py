class ModelContainer:
    def __init__(self, models=[]):#, default_num_iters=10, verbose_lvl=0):
        '''initializes model list and dicts'''
        self.models = models
        self.best_model = None
        self.predictions = None
        self.mean_recall_macro = {}
    #self.default_num_iters = default_num_iters
    #self.verbose_lvl = verbose_lvl
    
    def add_model(self, model):
        self.models.append(model)
    
    #    def scores(y_test, y_pred):
    #        """
    #            function that provides the metrics for the  multiclass classfiers
    #        """
    #
    #        precision, recall, fscore, support = score(y_test, y_pred)
    #
    #        print('precision: {}'.format(precision))
    #        print('recall: {}'.format(recall))
    #        print('fscore: {}'.format(fscore))
    #        print('support: {}'.format(support))
    
    def cross_validate(self, training_data_df, k=3, num_procs=1):
        '''cross validate models using given data'''
        feature_df = training_data_df.closest_cities
        target_df = training_data_df.sentiment
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # one hot encoding of the categorical features(cities)
        encoder = LabelBinarizer()
        X_train_lb = encoder.fit_transform(X_train)
        
        # oversample the minority class
        smote=SMOTE('minority')
        X_sm, Y_sm = smote.fit_sample(X_train_lb, y_train)
        
        for model in self.models:
            print(model)
            recall_macro = cross_val_score(model, X_sm, Y_sm, cv=k, n_jobs=num_procs, scoring='recall_macro')
            self.mean_recall_macro[model] = np.mean(recall_macro)
    

    def select_best_model(self):
        '''select model with lowest mse'''
        self.best_model = max(self.mean_recall_macro, key=self.mean_recall_macro.get)
    
    def best_model_fit(self, features, targets):
        '''fits best model'''
        self.best_model.fit(features, targets)
    
    def best_model_predict(self, features):
        '''scores features using best model'''
        self.predictions = self.best_model.predict(features)
    
    def print_summary(self):
        """
            prints summary of models, best model, and feature importance
            """
        print('\nModel Summaries:\n')
        for model in models.mean_recall_macro:
            print('\n', model, '- MSE:', models.mean_recall_macro[model])
        print('\nBest Model:\n', models.best_model)
        print('\nRECALL of Best Model\n', models.mean_recall_macro[models.best_model])
    
    def save_results(self):
        pass
