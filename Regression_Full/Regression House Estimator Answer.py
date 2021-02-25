import warnings
warnings.filterwarnings("ignore")


# ctrl + /
''' ==================== imports ==================== '''
# from ml_utils import * # Helper libraries
# import pickle # saving the model
''' ================================================= '''



''' =================== Load Data =================== '''
# dtf = pd.read_csv("data_houses.csv") # load 'data_houses.csv'
# cols = ["OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","FullBath","YearBuilt","YearRemodAdd",
#         "LotFrontage","MSSubClass"]
# dtf = dtf[["Id"]+cols+["SalePrice"]]
# # print(dtf.head())
# # dtf_overview(dtf, max_cat=20, figsize=(10,5))
''' ================================================= '''



''' =================== Format Data ================= '''
# dtf = dtf.set_index("Id") # Set the index column of the pandas data frame
# dtf = dtf.rename(columns={"SalePrice":"Y"})
# features = []
''' ================================================= '''



''' ================= Exploring Data ================ '''
# # What is the Average house price?
# freqdist_plot(dtf, "Y", box_logscale=True, figsize=(10,5)) # Mean house 181000

# #--- OverallQual ---#
# bivariate_plot(dtf, x="OverallQual", y="Y", figsize=(5,3))
# features.append("OverallQual")

# #--- YearBuilt ---#
# bivariate_plot(dtf, x="YearBuilt", y="Y", figsize=(10,3))
# features.append("YearBuilt")

# #--- YearRemodAdd ---#
# bivariate_plot(dtf, x="YearRemodAdd", y="Y", figsize=(10,3))
# features.append("YearRemodAdd")
#
# #--- GrLivArea ---#
# features.append("GrLivArea")
#
# #--- FullBath ---#
# features.append("FullBath")
#
#
# #--- LotFrontage ---#
# features.append("LotFrontage")
#
#
# # #--- GarageCars ---#
# features.append("GarageCars")
#
# # #--- GarageArea ---#
# features.append("GarageArea")
#
#
# #--- TotalBsmtSF, ---#
# features.append("TotalBsmtSF")
''' ================================================= '''



''' ========= Split Data Into Train and Test ======== '''
# check = data_preprocessing(dtf, y="Y", task="regression")
# dtf_train, dtf_test = dtf_partitioning(dtf, y="Y", test_size=0.3, shuffle=False) # Split into training and test 70/30
''' ================================================= '''



''' ============ Fill in Missing Values ============= '''
# dtf_train, lotfront_mean = fill_na(dtf_train, x="LotFrontage")
# dtf_test = fill_na(dtf_test, x="LotFrontage", value=lotfront_mean)
''' ================================================= '''



''' =========== Scaling Numerical Values ============ '''
# scalerX = preprocessing.RobustScaler(quantile_range=(25.0,75.0)) # Scale data to quantile range 25, 75
# scalerY = preprocessing.RobustScaler(quantile_range=(25.0,75.0)) # Scale data to quantile range 25, 75
#
# dtf_train, scalerX, scalerY = scaling(dtf_train, y="Y", scalerX=scalerX, scalerY=scalerY, task="regression")
# dtf_test = dtf_test[dtf_train.columns]
# dtf_test, _, _ = scaling(dtf_test, y="Y", scalerX=scalerX, scalerY=scalerY, fitted=True, task="regression")
# # dtf_overview(dtf_test)
''' ================================================= '''



''' ============= Observe Correlations ============== '''
#--- correlations ---#
# corr = corr_matrix(dtf_train, method="pearson", negative=False, annotation=True, figsize=(15,7))

#--- selection ---#
# X_names = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'YearBuilt', 'GarageArea'] # choose the most correlated columns
#
# X_train = dtf_train[X_names].values
# y_train = dtf_train["Y"].values
#
# X_test = dtf_test[X_names].values
# y_test = dtf_test["Y"].values
'''  ================================================= '''



''' ========== Initiate and Train model ============= '''
# model = ensemble.GradientBoostingRegressor()
#
# param_dic = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],      # weighting factor for the corrections by new trees when added to the model
#              'n_estimators':[100,250,500,750,1000,1250,1500,1750],  # number of trees added to the model
#              'max_depth':[2,3,4,5,6,7],                             # maximum depth of the tree
#              'min_samples_split':[2,4,6,8,10,20,40,60,100],         # sets the minimum number of samples to split
#              'min_samples_leaf':[1,3,5,7,9],                        # the minimum number of samples to form a leaf
#              'max_features':[2,3,4,5,6,7],                          # square root of features is usually a good starting point
#              'subsample':[0.7,0.75,0.8,0.85,0.9,0.95,1]}            # the fraction of samples to be used for fitting the individual base learners. Values lower than 1 generally lead to a reduction of variance and an increase in bias.
#
# model = tune_regr_model(X_train, y_train, model, param_dic, scoring="r2",searchtype="RandomSearch", n_iter=1000, cv=5, figsize=(10,5)) # set the model size
# pickle.dump(model, open('model_save.p', 'wb'))
# # model = pickle.load(open('model_save.p', 'rb'))
''' ================================================= '''



''' ============= Evaluate The model ================ '''
# model, predicted = fit_ml_regr(model, X_train, y_train, X_test, scalerY) # Predict the test data

# evaluate_regr_model(y_test, predicted, figsize=(25,5)) # Evaluate the predictions
# i = 1
# print("True:", "{:,.0f}".format(y_test[i]), "--> Pred:", "{:,.0f}".format(predicted[i]))
''' ================================================= '''



''' ============= Visualize The model ================ '''
# model2d = linear_model.LinearRegression()
# plot3d_regr_model(X_train, y_train, X_test, y_test, scalerY, model2d, rotate=(30,0), figsize=(7,5))
''' ================================================= '''
