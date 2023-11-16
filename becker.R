library(tidyverse)
library(vroom)
library(tidymodels)
library(timetk)
library(ggpubr)

train <- vroom("./train.csv")

test <- vroom("./test.csv")

## ML for Time Series
store3_item25 <- train %>%
  filter(store==3, item==25)

recipe <- recipe(sales~date, data=train) %>%
  step_date(date, features="doy") %>%
  step_date(date, features="dow")

prep <- prep(recipe)
baked_train <- bake(prep, new_data=train)

rf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees=1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Create a workflow with model & recipe

rf_workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(rf_model)

# Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range=c(1,(ncol(train) - 1))),
                            min_n(),
                            levels = 1)

# Set up K-fold CV
folds <- vfold_cv(train, v = 1, repeats=1)

CV_results <- rf_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(smape))

# Predict
bestTune <- CV_results %>%
  select_best("smape")

collect_metrics(CV_results) %>%
  filter(bestTune) %>%
  pull(mean)
