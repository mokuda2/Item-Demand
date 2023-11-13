library(tidyverse)
library(vroom)
library(tidymodels)
library(timetk)
library(ggpubr)

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

ggarrange(plot1, plot2, plot3, plot4)
