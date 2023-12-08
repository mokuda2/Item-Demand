library(tidyverse)
library(vroom)
library(tidymodels)
library(timetk)
library(ggpubr)
library(modeltime)
library(forecast)

train <- vroom("./STAT 348/Item-Demand/train.csv")
train

test <- vroom("./STAT 348/Item-Demand/test.csv")
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

# Cross-validate to tune model
cv_results <- modeltime_calibrate(es_model,
                                  new_data = testing(cv_split))

# Visualize CV results
p1 <- cv_results %>%
  modeltime_forecast(
                     new_data = testing(cv_split),
                     actual_data = store3_item26
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

# Evaluate the accuracy
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

# Cross-validate to tune model
cv_results2 <- modeltime_calibrate(es_model2,
                                  new_data = testing(cv_split))

# Visualize CV results
p3 <- cv_results2 %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = store3_item26
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

# Evaluate the accuracy
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

## ARIMA
store3_item25_train <- train %>%
  filter(store==3, item==25)

store3_item25_test <- test %>%
  filter(store==3, item==25)

recipe <- recipe(sales~date, data=store3_item25_train) %>%
  step_date(date, features="doy") %>%
  step_date(date, features="dow") %>%
  step_date(date, features="month")

prep <- prep(recipe)
baked_train <- bake(prep, new_data=store3_item25_train)

arima_model <- arima_reg(seasonal_period=365,
                         non_seasonal_ar=5, # default max p to tune
                         # non_seasona_ma=5, # default max q to tune
                         seasonal_ar=2, # default max P to tune
                         seasonal_ma=2, #default max Q to tune
                         non_seasonal_differences=2, # default max d to tune
                         seasonal_differences=2 #default max D to tune
  ) %>%
  set_engine("auto_arima")

cv_split <- time_series_split(store3_item25_train, assess="3 months", cumulative = TRUE)

cv_split %>%
  tk_time_series_cv_plan() %>% #Put into a data frame
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

arima_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(arima_model) %>%
  fit(data=training(cv_split))

cv_results <- modeltime_calibrate(arima_wf,
                                  new_data = testing(cv_split))

# Visualize CV results
a1 <- cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = store3_item25_train
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

# Evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

# Refit to all data then forecast
es_fullfit <- cv_results %>%
  modeltime_refit(data = store3_item25_train)

# es_preds <- es_fullfit %>%
#   modeltime_forecast(h = "3 months") %>%
#   rename(date=.index, sales=.value) %>%
#   select(date, sales) %>%
#   full_join(., y=test, by="date") %>%
#   select(id, sales)

a2 <- es_fullfit %>%
  modeltime_forecast(new_data = store3_item25_test, actual_data = store3_item25_train) %>%
  plot_modeltime_forecast(.interactive=FALSE)

# part 2
store3_item26_train <- train %>%
  filter(store==3, item==26)

store3_item26_test <- test %>%
  filter(store==3, item==26)

recipe2 <- recipe(sales~date, data=store3_item26_train) %>%
  step_date(date, features="doy") %>%
  step_date(date, features="dow") %>%
  step_date(date, features="month")

prep2 <- prep(recipe2)
baked_train <- bake(prep2, new_data=store3_item26_train)

cv_split2 <- time_series_split(store3_item26_train, assess="3 months", cumulative = TRUE)
cv_split2 %>%
  tk_time_series_cv_plan() %>% #Put into a data frame
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

arima_wf2 <- workflow() %>%
  add_recipe(recipe2) %>%
  add_model(arima_model) %>%
  fit(data=training(cv_split2))

# Cross-validate to tune model
cv_results2 <- modeltime_calibrate(arima_wf2,
                                   new_data = testing(cv_split2))

# Visualize CV results
a3 <- cv_results2 %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = store3_item26_train
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
  modeltime_refit(data = store3_item26_train)

# es_preds2 <- es_fullfit2 %>%
#   modeltime_forecast(h = "3 months") %>%
#   rename(date=.index, sales=.value) %>%
#   select(date, sales) %>%
#   full_join(., y=test, by="date") %>%
#   select(id, sales)

a4 <- es_fullfit2 %>%
  modeltime_forecast(new_data = store3_item26_test, actual_data = store3_item26_train) %>%
  plot_modeltime_forecast(.interactive=FALSE)

image <- plotly::subplot(a1, a3, a2, a4, nrows=2)

## prophet
store3_item25_train <- train %>%
  filter(store==3, item==25)

store3_item25_test <- test %>%
  filter(store==3, item==25)

recipe <- recipe(sales~date, data=store3_item25_train) %>%
  step_date(date, features="doy") %>%
  step_date(date, features="dow") %>%
  step_date(date, features="month")

prep <- prep(recipe)
baked_train <- bake(prep, new_data=store3_item25_train)

cv_split <- time_series_split(store3_item25_train, assess="3 months", cumulative = TRUE)

prophet_model <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(sales ~ date, data = training(cv_split))

cv_results <- modeltime_calibrate(prophet_model,
                                  new_data = testing(cv_split))

# Visualize CV results
b1 <- cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = store3_item25_train
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

es_fullfit <- cv_results %>%
  modeltime_refit(data = store3_item25_train)

b2 <- es_fullfit %>%
  modeltime_forecast(new_data = store3_item25_test, actual_data = store3_item25_train) %>%
  plot_modeltime_forecast(.interactive=FALSE)

# part 2
store3_item26_train <- train %>%
  filter(store==3, item==26)

store3_item26_test <- test %>%
  filter(store==3, item==26)

recipe2 <- recipe(sales~date, data=store3_item26_train) %>%
  step_date(date, features="doy") %>%
  step_date(date, features="dow") %>%
  step_date(date, features="month")

prep2 <- prep(recipe2)
baked_train <- bake(prep2, new_data=store3_item26_train)

cv_split2 <- time_series_split(store3_item26_train, assess="3 months", cumulative = TRUE)

prophet_model2 <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(sales ~ date, data = training(cv_split2))

# Cross-validate to tune model
cv_results2 <- modeltime_calibrate(prophet_model2,
                                   new_data = testing(cv_split2))

# Visualize CV results
b3 <- cv_results2 %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = store3_item26_train
  ) %>%
  plot_modeltime_forecast(.interactive=TRUE)

# Refit to all data then forecast
es_fullfit2 <- cv_results2 %>%
  modeltime_refit(data = store3_item26_train)

preds <- es_fullfit2 %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>% 
  full_join(., y=test, by="date") %>%
  select(id, sales)

test

b4 <- es_fullfit2 %>%
  modeltime_forecast(new_data = store3_item26_test, actual_data = store3_item26_train) %>%
  plot_modeltime_forecast(.interactive=FALSE)

image <- plotly::subplot(b1, b3, b2, b4, nrows=2)

