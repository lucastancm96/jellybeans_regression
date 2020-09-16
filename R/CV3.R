# Version: Model 3 (Without Nov)
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

########## Create Model 3 (Without Nov) ##########
mod3 <- lm(sales ~ tv 
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
           + Dec, data=df2)
summary(mod3)

########## Cross Validation Starts Here ##########
# 1. Split data into train and test set based on 80:20
Use default data
# 2. Perform variable selection
# Forward
mod3fwd <- ols_step_forward_p(mod3)
mod3fwd
plot(mod3fwd)
# Variable removed: Mar, Jul, Aug, Oct
# R2: 0.982
# Adj R2: 0.980
# RMSE: 291.306

# Stepwise
mod3stpw <- ols_step_both_p(mod3)
mod3stpw
plot(mod3stpw)
# Variable removed: Mar,Jul, Aug, Oct
# R2 = 0.982
# Adj R2 = 0.980
# RMSE = 291.306

### Both variable selection shows the same results.

# 3. Create new model after variable selection
new_mod3 <- lm(sales ~ region 
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
summary(new_mod3)
# Summary shows Dec is insignificant. Hence, remove it.


# 4. Performing CV on training model
cv_mod3 <- cv.lm(data=df2_trainingset,
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
                                     m = 144 ,plotit = FALSE) 

# 5. Calculate RMSE for training model
mod3_trainMSE <- sum( (cv_mod3[,"sales"]-cv_mod3[,"cvpred"])^2  )  / nrow(df2_trainingset)
mod3_trainMSE
mod3_trainRMSE <- sqrt(mod3_trainMSE)
mod3_trainRMSE

trc <- trainControl(method = 'LOOCV')
mod3_trainMSEtrc <- train(sales ~ region 
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
mod3_trainRMSEtrc <- mod3_trainMSEtrc
mod3_trainRMSEtrc
# RMSE for train_MSPE = 311

# 6. Create final model after cross validation
final_mod3 <- lm(sales ~ region 
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
mod3_testpredict <- predict.lm(object = final_mod3,newdata = df2_testset)
# 8. Calculating RMSE for test set
mod3_testMSE = sum( (df2_testset[,"sales"]-mod3_testpredict)^2 ) / nrow(df2_testset)
mod3_testMSE
mod3_testRMSE <- sqrt(mod3_testMSE)
mod3_testRMSE
# RMSE for test_MSPE = 325

### RESULT SHOWS THAT USING MODEL AFTER VARIABLE SELECTION, THE RMSE ARE:
### TRAIN = 311
### TEST = 325