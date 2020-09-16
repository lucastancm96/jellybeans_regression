# Version: Model 4 (Without Dec)
#Import all library
library(readxl)
require(ggplot)
require(randomForest)
require(DAAG)
require(lmtest)
require(caret)
require(olsrr)
require(car)

# Import dataset
df <- read_xls('Data_JellyBeansStudent_6.xls')

# Rename ad1 & ad2
names(df)[4:5] <- c('tv','banner')
View(df)

# Convert 'time' into year
df$year1 <- df$time >= 1 & df$time <= 12
df$year1 <- as.numeric(df$year1)

df$year2 <- df$time >= 13 & df$time <= 24
df$year2 <- as.numeric(df$year2)

df$year3 <- df$time >= 25 & df$time <= 36
df$year3 <- as.numeric(df$year3)

years <- c('year1', 'year2', 'year3')

df[years] <- lapply(df[years], factor)

df = subset(df, select = -c(time))

# Create dummy variable for month
df2 <- cbind(df)
df2$Jan <- as.numeric(df2$month == 'Jan')
df2$Feb <- as.numeric(df2$month == 'Feb')
df2$Mar <- as.numeric(df2$month == 'Mar')
df2$Apr <- as.numeric(df2$month == 'Apr')
df2$May <- as.numeric(df2$month == 'May')
df2$Jun <- as.numeric(df2$month == 'Jun')
df2$Jul <- as.numeric(df2$month == 'Jul')
df2$Aug <- as.numeric(df2$month == 'Aug')
df2$Sep <- as.numeric(df2$month == 'Sep')
df2$Oct <- as.numeric(df2$month == 'Oct')
df2$Nov <- as.numeric(df2$month == 'Nov')
df2$Dec <- as.numeric(df2$month == 'Dec')

month <- c('Jan', 'Feb', 'Mar',
           'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep',
           'Oct', 'Nov', 'Dec')

df2[month] <- lapply(df2[month], factor)
df2 = subset(df2, select = -c(product))

########## Create Model 4 (Without Dec) ##########
mod4 <- lm(sales ~ tv 
           + banner 
           + prom 
           + price 
           + region 
           + Jan 
           + Feb 
           + Mar 
           + Apr 
           + May 
           + Jun 
           + Jul 
           + Aug 
           + Sep 
           + Oct 
           + Nov, data=df2)
summary(mod4)

########## Cross Validation Starts Here ##########
# 1. Split data into train and test set based on 80:20

# 2. Perform variable selection
# Forward
mod4fwd <- ols_step_forward_p(mod4)
mod4fwd
plot(mod4fwd)
# Variable removed: Aug, Nov
# R2: 0.982
# Adj R2: 0.980
# RMSE: 290.913

# Stepwise
mod4stpw <- ols_step_both_p(mod4)
mod4stpw
plot(mod4stpw)
# Variable removed: Mar, Jul, Aug, Oct, Nov
# R2 = 0.981
# Adj R2 = 0.980
# RMSE = 291.967

# Both forward and stepwise shows different removal of variable. Hence, CV both.

# 3. Create new model after variable selection
new_mod4fwd <- lm(sales ~ region 
               + Apr 
               + prom 
               + tv
               + price
               + banner
               + May
               + Jun
               + Sep
               + Jan
               + Feb, 
               data=df2)
summary(new_mod4fwd)
# Summary shows Jul, Oct and Mar are insignificant. Hence, remove it.

new_mod4stpw <- lm(sales ~ region 
                  + Apr 
                  + prom 
                  + tv
                  + price
                  + banner
                  + May
                  + Jun
                  + Sep
                  + Jan
                  + Feb, 
                  data=df2)
summary(new_mod4stpw)

# After removal of additional variables on new_mod4fwd, it becomes the same as new_mod4stpw now.
# Hence, use only one model.
new_mod4 <- lm(sales ~ region 
                   + Apr 
                   + prom 
                   + tv
                   + price
                   + banner
                   + May
                   + Jun
                   + Sep
                   + Jan
                   + Feb, 
                   data=df2)
summary(new_mod4)

# 4. Performing CV on training model
cv_mod4 <- cv.lm(data=df2_trainingset,
                 form.lm=formula(sales ~ region 
                                 + Apr 
                                 + prom 
                                 + tv
                                 + price
                                 + banner
                                 + May
                                 + Jun
                                 + Sep
                                 + Jan
                                 + Feb),
                 m = 126 ,plotit = FALSE) 

# 5. Calculate RMSE for training model
mod4_trainMSE <- sum( (cv_mod4[,"sales"]-cv_mod4[,"cvpred"])^2  )  / nrow(df2_trainingset)
mod4_trainMSE
mod4_trainRMSE <- sqrt(mod4_trainMSE)
mod4_trainRMSE

trc <- trainControl(method = 'LOOCV')
mod4_trainRMSEtrc <- train(sales ~ region 
                          + Apr 
                          + prom 
                          + tv
                          + price
                          + banner
                          + May
                          + Jun
                          + Sep
                          + Jan
                          + Feb,
                          data = df2_trainingset,
                          method = 'lm',
                          trControl = trc)
mod4_trainRMSEtrc 
# RMSE for train_MSPE = 312

# 6. Create final model after cross validation
final_mod4 <- lm(sales ~ region 
                 + Apr 
                 + prom 
                 + tv
                 + price
                 + banner
                 + May
                 + Jun
                 + Sep
                 + Jan
                 + Feb,
                 data = df2_trainingset)

# 7. CV test set with training model
mod4_testpredict <- predict.lm(object = final_mod4,newdata = df2_testset)
# 8. Calculating RMSE for test set
mod4_testMSE = sum( (df2_testset[,"sales"]-mod4_testpredict)^2 ) / nrow(df2_testset)
mod4_testMSE
mod4_testRMSE <- sqrt(mod4_testMSE)
mod4_testRMSE
# RMSE for test_MSPE = 294

### RESULT SHOWS THAT USING MODEL AFTER VARIABLE SELECTION, THE RMSE ARE:
### TRAIN = 312
### TEST = 294

# 2nd random dataset
### TRAIN = 335
### TEST = 255

# 3rd random dataset
### TRAIN = 311
### TEST = 325
