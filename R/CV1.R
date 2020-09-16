# Version: Model 1 (Without Dummy)
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

########## Create Model 1 (Without Dummy) ##########
mod1 <- lm(sales ~ tv 
           + banner 
           + prom 
           + price 
           + region 
           + month, data=df2)
summary(mod1)

########## Cross Validation Starts Here ##########
# 1. Split data into train and test set based on 80:20
n = nrow(df)
df_trainingsetsize <- round(n*.7)
df_randomselection <- sample(x=1:nrow(df), size=df_trainingsetsize)
df_trainingset <- df[df_randomselection,]
df_testset <- df[-df_randomselection,]
View(df_testset)
# 2. Perform variable selection
# Forward
mod1fwd <- ols_step_forward_p(mod1)
mod1fwd
plot(mod1fwd)
# Variable removed: Jan, Feb, Sep
# R2: 0.982
# Adj R2: 0.980
# RMSE: 290.913

# Stepwise
mod1stpw <- ols_step_both_p(mod1)
mod1stpw
plot(mod1stpw)
# Variable removed: Jan, Feb, Mar, Jul, Sep, Oct
# R2 = 0.981
# Adj R2 = 0.980
# RMSE = 291.967

# Both forward and stepwise shows the same results. Hence, use one model.

# 3. Create new model after variable selection
new_mod1 <- lm(sales ~ tv 
               + banner 
               + prom 
               + price 
               + region 
               + month, 
                  data=df2)
summary(new_mod1)
# No insignificant variables.

# 4. Performing CV on training model
cv_mod1 <- cv.lm(data=df_trainingset,
                    form.lm=formula(sales ~ tv 
                                    + banner 
                                    + prom 
                                    + price 
                                    + region 
                                    + month),
                    m = 126 ,plotit = FALSE) 

# 5. Calculate RMSE for training model
mod1_trainMSE <- sum( (cv_mod1[,"sales"]-cv_mod1[,"cvpred"])^2  )  / nrow(df_trainingset)
mod1_trainMSE
mod1_trainRMSE <- sqrt(mod1_trainMSE)
mod1_trainRMSE

trc <- trainControl(method = 'LOOCV')
mod1_trainRMSEtrc <- train(sales ~ tv 
                              + banner 
                              + prom 
                              + price 
                              + region 
                              + month,
                              data = df_trainingset,
                              method = 'lm',
                              trControl = trc)
mod1_trainRMSEtrc 
# RMSE for mod2fwd_trainRMSE = 317

# 6. Create final model after cross validation
final_mod1 <- lm(sales ~ tv 
                 + banner 
                 + prom 
                 + price 
                 + region 
                 + month,
                 data = df_trainingset)

# 7. CV test set with training model
mod1_testpredict <- predict.lm(object = final_mod1,newdata = df_testset)

# 8. Calculating RMSE for test set
mod1_testMSE = sum( (df_testset[,"sales"]-mod1_testpredict)^2 ) / nrow(df_testset)
mod1_testMSE
mod1_testRMSE <- sqrt(mod1_testMSE)
mod1_testRMSE
# RMSE for mod2fwd_testRMSE = 345

### RESULT SHOWS THAT USING MODEL AFTER VARIABLE SELECTION, THE RMSE ARE:
### TRAIN = 317
### TEST  = 345
