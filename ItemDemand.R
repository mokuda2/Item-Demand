library(tidyverse)
library(vroom)
library(tidymodels)
library(timetk)
library(ggpubr)
library(modeltime)
library(orca)

train <- vroom("./Item-Demand/train.csv")
train

test <- vroom("./Item-Demand/test.csv")
test

item1 <- train[train$item == 1 & train$store == 1,]
item2 <- train[train$item == 2 & train$store == 1,]
item3 <- train[train$item == 3 & train$store == 1,]
item4 <- train[train$item == 4 & train$store == 1,]

item_list = c(item1, item2, item3, item4)

par(mfrow = c(2, 2))

plot1 <- item1 %>%
  pull(sales) %>%
  forecast::ggAcf(.)

plot2 <- item2 %>%
  pull(sales) %>%
  forecast::ggAcf(.)

plot3 <- item3 %>%
  pull(sales) %>%
  forecast::ggAcf(.)

plot4 <- item4 %>%
  pull(sales) %>%
  forecast::ggAcf(.)

plots <- ggarrange(plot1, plot2, plot3, plot4)
ggsave("./Item-Demand/acf.png", plot = plots)

## ML for Time Series
store3_item25 <- train %>%
  filter(store==3, item==25)

recipe <- recipe(sales~date, data=store3_item25) %>%
  step_date(date, features="doy") %>%
  step_date(date, features="dow") %>%
  step_date(date, features="month")

prep <- prep(recipe)
baked_train <- bake(prep, new_data=store3_item25)

rf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees=500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Create a workflow with model & recipe

rf_workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(rf_model)

# Set up grid of tuning values
tuning_grid <- grid_regular(mtry(range=c(1,(ncol(train) - 1))),
                            min_n(),
                            levels = 5)

# Set up K-fold CV
folds <- vfold_cv(store3_item25, v = 5, repeats=1)

CV_results <- rf_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(smape))

# Predict
bestTune <- CV_results %>%
  select_best("smape")

collect_metrics(CV_results) %>%
  filter(mtry == bestTune$mtry, min_n == bestTune$min_n, .config == bestTune$.config) %>%
  pull(mean)

## exponential smoothing
store3_item25 <- train %>%
  filter(store==3, item==25)
cv_split <- time_series_split(store3_item25, assess="3 months", cumulative = TRUE)
cv_split %>%
  tk_time_series_cv_plan() %>% #Put into a data frame
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

es_model <- exp_smoothing() %>%
  set_engine("ets") %>%
  fit(sales~date, data=training(cv_split))

## Cross-validate to tune model
cv_results <- modeltime_calibrate(es_model,
                                  new_data = testing(cv_split))

## Visualize CV results
p1 <- cv_results %>%
  modeltime_forecast(
                     new_data = testing(cv_split),
                     actual_data = store3_item26
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
                           .interactive = FALSE
  )

# Refit to all data then forecast
es_fullfit <- cv_results %>%
  modeltime_refit(data = store3_item25)

es_preds <- es_fullfit %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)

p2 <- es_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = store3_item25) %>%
  plot_modeltime_forecast(.interactive=FALSE)

# part 2
store3_item26 <- train %>%
  filter(store==3, item==26)
cv_split2 <- time_series_split(store3_item26, assess="3 months", cumulative = TRUE)
cv_split2 %>%
  tk_time_series_cv_plan() %>% #Put into a data frame
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

es_model2 <- exp_smoothing() %>%
  set_engine("ets") %>%
  fit(sales~date, data=training(cv_split))

## Cross-validate to tune model
cv_results2 <- modeltime_calibrate(es_model2,
                                  new_data = testing(cv_split))

## Visualize CV results
p3 <- cv_results2 %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = store3_item26
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cv_results2 %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

# Refit to all data then forecast
es_fullfit2 <- cv_results2 %>%
  modeltime_refit(data = store3_item26)

es_preds2 <- es_fullfit2 %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)

p4 <- es_fullfit2 %>%
  modeltime_forecast(h = "3 months", actual_data = store3_item26) %>%
  plot_modeltime_forecast(.interactive=FALSE)

image <- plotly::subplot(p1, p3, p2, p4, nrows=2)
# orca(image, "./Item-Demand/image.png")
