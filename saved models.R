# Random Forest Model
set.seed(777)
rf_mod <- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("regression")

rf_wf <- workflow() %>%
  add_recipe(airbnb_recipe) %>%
  add_model(rf_mod)
  # Grid
param_grid <- grid_regular(mtry(range = c(6,28)), 
                           trees(range = c(2, 5)), 
                           min_n(), levels = c(10,10,10))
  # Fold
airbnb_fold <- vfold_cv(final_train, v = 5, repeates = 5, strata = price)

set.seed(777)
  # tuned model
tune_res_rf <- tune_grid(
  rf_wf, 
  resamples = airbnb_fold, 
  grid = param_grid,
)


# Boosted Tree
set.seed(777)
boost_spec <- boost_tree(trees = tune(), tree_depth = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

boost_wf <- workflow() %>% 
  add_recipe(airbnb_recipe) %>% 
  add_model(boost_spec)
   # grid
boost_grid <- grid_regular(trees(range = c(50, 350)), 
                           tree_depth(), levels = c(4,8))
   # tuned model
tune_res_boost <- tune_grid(
  boost_wf,
  resamples = airbnb_fold, 
  grid = boost_grid
)

save(tune_res_rf,tune_res_boost, file = "tunedmodels.rda")
rm(tune_res_tree,tune_res_rf,tune_res_boost)
load(file = "tunedmodels.rda")


?roc_auc
