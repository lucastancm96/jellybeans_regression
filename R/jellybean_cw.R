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

# Check type of data. For chr, can straightaway add into the model; 
# for categorical var in num (e.g. region, month), convert into categorical var using as.factor;
# for 1 level categorical var (e.g. product), no need add into model #contrasts can be applied only to factors with 2 or more levels
str(df)

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

str(df)

########## Model 1: Multiple Regression Model (Marketing & Non-Marketing Variable) ########## 

mod1 <- lm(sales ~ tv + banner + prom + price + region + month, data=df)
summary(mod1)

# Interpretation
# 1. F-statistic p-value < 0.05, highly significant. 
# At least one predictor variables is significantly related to the sales.
# 2. R2 = 0.982 means 98.20% of the variance in sales can be predicted by 
# all the marketing and non-marketing variables in the model.
# Calculate error rate
sigma(mod1)/mean(df$sales)
# 3. RSE of 293 corresponding to 7.14% of error rate.
# 4. t-statistic p-value for tv, banner and prom < 0.05, hence they have significant association with sales
# 5. For every increase of 1 GRP in TV, 10.748 packs will be sold
# 6. For every increase of 1 GRP in Banner, 24.514 packs will be sold
# 7. For every increase of 1% of stores with Promotion, 30.571 packs will be sold
# 8. When compare other regions to Capital, it shows that Capital has relatively higher sales 
# 9. When compare other months to Capital, it shows that April has relatively higher sales

# Find correlation between sales and other variables
cor(df[,2:6], use = 'pairwise.complete.obs', method='pearson')

# Finding association between sales & marketing variables(TV, Banner & Prom)
sales_tv <- cor.test(df$tv, df$sales, method='pearson')
sales_tv
# Correlation between sales and tv is 0.171, indicate a weak positive correlation/association.

sales_banner <- cor.test(df$banner, df$sales, method='pearson')
sales_banner
# Correlation between sales and banner is 0.02, indicate a very weak positive correlation/association.

sales_prom <- cor.test(df$prom, df$sales, method='pearson')
sales_prom
# Correlation between sales and prom is 0.5, indicate a moderate positive orrelation/association.
# Hence, all marketing variables has positive association with sales.

# Calculate total predicted packs sold based on Model 1
sum(predict(mod1)) # 736992

# How many packs sold associated to advertisement?
# 1. TV
tv_sales <- mod1$coefficients[2]*sum(df$tv)
# 6390 * 10.748 = 68680
print(tv_sales)
# Total packs sold associated with 6390 GRP of TV = 68680

# 2. Banner
banner_sales <- mod1$coefficient[3]*sum(df$banner)
# 1465 * 24.514
print(banner_sales)
# Total packs sold associated with 1465 GRP of banner = 35913

# 3. Prom
prom_sales <- mod1$coefficient[4]*sum(df$prom)
# 3945 * 30.571
print(prom_sales)
# Total packs sold associated with 3945 percentage of stores = 120603

# Calculate the total sales contributed by advertisement and promotion
sum(tv_sales, banner_sales)
# ads = 104593
# prom = 120603

# Calculate the sales that are not contributed by advertisement and promotion
sum(df$sales) - tv_sales - banner_sales - prom_sales
# Sales contributed by non-marketing variable = 511796

# Calculate the efficiency of TV and Banner.
# TV
2000000/tv_sales
# The company need to spend £29.1 in order to get 1 pack sold

# Banner
500000/banner_sales
# The company need to invest £13.9 in order to get 1 pack sold
# Hence, banner is more efficient than tv

########## Plotting Model 1 ##########
par(mfrow=c(2,2))
plot(mod1)

########## Perform Assumption Checking ##########
# Perform Statistical test to check the presence of heteroscedasticity
# H0 = variance of the residual is constant 
# H0 = homoscedasticity exists
# H1 = variance of the residual is not constant
# H1 = heterscedasticity exists

lmtest::bptest(mod1) 
#bp test shows p-value = 0.08 > 0.05, fail to reject H0, not enough evidence to claim non-constant variance of our residual errors
car::ncvTest(mod1)
#ncv test shows p-value = 0.5 > 0.05, fail to reject H0, not enough evidence to claim non-constant variance of our residual errors

# Perform Statistical test to check if residual are normally distributed
# H0 = residual errors distribution is not statistically different from a normal distribution
# H1 = residual errors distribution are statistically different from a normal distribution
ols_test_normality(mod1) 
# Shapiro-Wilk returns a p-value of 0.3050 > 0.05
# Kolmogorov-Smirnov test returns a p-value of 0.9642 > 0.05
# Anderson-Darling returns a p-value of 0.3050 > 0.05
# Hence, fail to reject H0. Residual is not statistically different from a normal distribution (Residual is normally distributed)

# Autocorrelation test
# H0 = independent variables are not positively correlated with each other
# H0 = autocorrelation does not exists
# H1 = positive correlation exists between independent variables
# H1 = autocorrelation exists
dwtest(mod1) 
# p-value = 0.009 < 0.05, reject H0, can claim autocorrelation exists 

########## Finding other possible source of the variation in sales other than marketing variables

########## Cross Validation Starts Here ##########
# 1. Split data into train and test set
set.seed(123)
n = nrow(df)
df_trainingsetsize <- round(n*.7)
df_randomselection <- sample(x=1:nrow(df2), size=df_trainingsetsize)
df_trainingset <- df[df_randomselection,]
df_testset <- df[-df_randomselection,]

# 2. Perform variable selection
# Stepwise Selection
mod1_slc3 <- ols_step_both_p(mod1)
mod1_slc3
plot(mod1_slc3)
# RESULT:
# R2 = 0.982
# Adj R2 = 0.980
# RMSE = 292.526
# No variable removed

# 3. Create new model after variable selection
mod1_stpw <- lm(sales ~ region
               + month
               + prom
               + tv
               + price
               + banner
               , data=df2)
summary(mod1_stpw)
# 4. Performing CV on training model
# mod1
cv_mod1 <- cv.lm(data=df_trainingset,
                     form.lm=formula(sales ~ region
                                     + month
                                     + prom
                                     + tv
                                     + price
                                     + banner),
                     m = 144 ,plotit = FALSE) 

# 5. Calculate RMSE for training model
# mod1
trc <- trainControl(method = 'LOOCV')

mod1_train_RMSE <- sum( (cv_mod1[,"sales"]-cv_mod1[,"cvpred"])^2  )  / nrow(df_trainingset)
sqrt(mod1_train_RMSE)

mod1_train_RMSEtrc <- train(sales ~ region
                            + month
                            + prom
                            + tv
                            + price
                            + banner,
                          data = df_trainingset,
                          method = 'lm',
                          trControl = trc)
mod1_train_RMSEtrc
# RMSE for mod2_fwd = 325

# 6. Create final model after cross validation
# mod1
final_mod1 <- lm(sales ~ region
                     + month
                     + prom
                     + tv
                     + price
                     + banner,
                     data = df_trainingset)

# 7. CV test set with training model
mod1_test<- predict.lm(object = final_mod1,newdata = df_testset)

# Both also can use to calculate RMSE
# Calculating RMSE for test set
mod1_test_RMSE = sum( (df_testset[,"sales"]-mod1_test)^2 ) / nrow(df_testset)
sqrt(mod1_test_RMSE)
# RMSE for mod2_test_fwd = 298

########## Model 2: Multiple Regression Model (Marketing & Non-Marketing Variable, with Month in dummy variables) ########## 
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
str(df2)

########## Create Model 2 (With all dummy months retained) ##########
mod2 <- lm(sales ~ tv 
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
           + Nov 
           + Dec, data=df2)
summary(mod2)

# Interpretation
# 1. F-statistic p-value < 0.05, highly significant. 
# At least one predictor variables is significantly related to the sales.
# 2. R2 = 0.982 means 98.20% of the variance in sales can be predicted by 
# all the marketing and non-marketing variables in the model.
# Calculate error rate
sigma(mod2)/mean(df2$sales)
# 3. RSE of 293 corresponding to 7.14% of error rate.
# 4. t-statistic p-value for tv, banner and prom < 0.05, hence they have significant association with sales
# 5. For every increase of 1 GRP in TV, 10.748 packs will be sold
# 6. For every increase of 1 GRP in Banner, 24.514 packs will be sold
# 7. For every increase of 1% of stores with Promotion, 30.571 packs will be sold
# 8. When compare other regions to Capital, it shows that Capital has relatively higher sales 
# R2 and t-statisic of marketing var are same in both Model 1 & 2

# Show correlation between sales and variables
cor(df2[,2:6], use = 'pairwise.complete.obs', method='pearson')

# Finding association between sales & marketing variables(TV, Banner & Prom)
sales_tv <- cor.test(df2$tv, df2$sales, method='pearson')
sales_tv
# Correlation between sales and tv is 0.171, indicate a weak positive correlation/association.
sales_banner <- cor.test(df2$banner, df2$sales, method='pearson')
sales_banner
# Correlation between sales and banner is 0.02, indicate a very weak positive correlation/association.
sales_prom <- cor.test(df2$prom, df2$sales, method='pearson')
sales_prom
# Correlation between sales and prom is 0.5, indicate a moderate positive orrelation/association.
# Hence, all marketing variables has positive association with sales.

# Calculate total predicted packs sold based on Model 1
sum(predict(mod2)) # 736992

# How many packs sold associated to advertisement?
# 1. TV
tv_sales <- mod2$coefficients[2]*sum(df2$tv)
# 6390 * 10.748 = 68680
print(tv_sales)
# Total packs sold associated with 6390 GRP of TV = 68680

# 2. Banner
banner_sales <- mod2$coefficient[3]*sum(df2$banner)
# 1465 * 24.514
print(banner_sales)
# Total packs sold associated with 1465 GRP of banner = 35913

# 3. Prom
prom_sales <- mod2$coefficient[4]*sum(df2$prom)
# 3945 * 30.571
print(prom_sales)
# Total packs sold associated with 3945 percentage of stores = 120603

# Calculate the total sales contributed by advertisement and promotion
sum(tv_sales, banner_sales)
# ads = 104593
# prom = 120603

# Calculate the sales that are not contributed by advertisement and promotion
sum(df$sales) - tv_sales - banner_sales - prom_sales
# Sales contributed by non-marketing variable = 511796

# Calculate the efficiency of TV and Banner.
# TV
2000000/tv_sales
# The company need to spend £29.1 in order to get 1 pack sold

# Banner
500000/banner_sales
# The company need to invest £13.9 in order to get 1 pack sold
# Hence, banner is more efficient than tv

########## Plotting Model 2 ##########
par(mfrow=c(2,2))
plot(mod2)

##### Model 1 and Model 2 Plot are the same

########## Perform Assumption Checking ##########
# Perform Statistical test to check the presence of heteroscedasticity
# H0 = variance of the residual is constant 
# H0 = homoscedasticity exists
# H1 = variance of the residual is not constant
# H1 = heterscedasticity exists
lmtest::bptest(mod2) 
#bp test shows p-value = 0.08 > 0.05, fail to reject H0, not enough evidence to claim non-constant variance of our residual errors
car::ncvTest(mod2)
#ncv test shows p-value = 0.5 > 0.05, fail to reject H0, not enough evidence to claim non-constant variance of our residual errors

# Perform Statistical test to check if residual are normally distributed
# H0 = residual errors distribution is not statistically different from a normal distribution
# H1 = residual errors distribution are statistically different from a normal distribution
ols_test_normality(mod2) 
# Shapiro-Wilk returns a p-value of 0.3050 > 0.05
# Kolmogorov-Smirnov test returns a p-value of 0.9642 > 0.05
# Anderson-Darling returns a p-value of 0.3050 > 0.05
# Hence, fail to reject H0. Residual is not statistically different from a normal distribution (Residual is normally distributed)

# Autocorrelation test
# H0 = independent variables are not positively correlated with each other
# H0 = autocorrelation does not exist
# H1 = positive correlation exists between independent variables
# H1 = autocorrelation exist
dwtest(mod2)
# p-value = 0.7 > 0.05, cannot reject H0, independent variables are not positively correlated with each other
# autocorrelation does not exists

########## Finding other possible source of the variation in sales other than marketing variables

########## Cross Validation Starts Here ##########
# 1. Split data into train and test set
set.seed(123)
n = nrow(df2)
df2_trainingsetsize <- round(n*.7)
df2_randomselection <- sample(x=1:nrow(df2), size=df2_trainingsetsize)
df2_trainingset <- df2[df2_randomselection,]
df2_testset <- df2[-df2_randomselection,]

# 2. Perform variable selection
# Forward Selection
mod2_slc1 <- ols_step_forward_p(mod2)
mod2_slc1
plot(mod2_slc1)
# RESULT:
# R2 = 0.982
# Adj R2 = 0.980
# RMSE = 291.047
# Months removed = Jan, Feb, Sep

# Backward Selection
mod2_slc2 <- ols_step_backward_p(mod2)
mod2_slc2
plot(mod2_slc2)
##### No variables have been removed, hence discard this method

# Stepwise Selection
mod2_slc3 <- ols_step_both_p(mod2)
mod2_slc3
plot(mod2_slc3)
# RESULT:
# R2 = 0.981
# Adj R2 = 0.980
# RMSE = 293.812
# Months removed = Jan, Feb, Mar, Jul, Sep, Oct

# 3. Create new model after variable selection
mod2_fwd <- lm(sales ~ tv 
                     + banner 
                     + prom 
                     + price 
                     + region 
                     + Mar 
                     + Apr 
                     + May 
                     + Jun 
                     #+ Jul 
                     + Aug 
                     #+ Oct 
                     + Nov 
                     + Dec, data=df2)
summary(mod2_fwd)
# Summary shows hat Jul and Oct is not significant. Hence, remove it.
# Total variables removed: Jan, Feb, Sep, Jul, Oct
mod2_stpw <- lm(sales ~ tv 
                      + banner 
                      + prom 
                      + price 
                      + region 
                      + Apr 
                      + May 
                      + Jun 
                      + Aug 
                      + Nov 
                      + Dec, data=df2)
summary(mod2_stpw)

# 4. Performing CV on training model
# model2_fwd
cv_mod2_fwd <- cv.lm(data=df2_trainingset,
                           form.lm=formula(sales ~ tv 
                                           + banner 
                                           + prom 
                                           + price 
                                           + region 
                                           + Mar 
                                           + Apr 
                                           + May 
                                           + Jun 
                                           #+ Jul 
                                           + Aug 
                                           #+ Oct 
                                           + Nov 
                                           + Dec),
                           m = 126 ,plotit = FALSE) 

# model2_stpw
cv_mod2_stpw <- cv.lm(data=df2_trainingset,
                           form.lm=formula(sales ~ tv 
                                           + banner 
                                           + prom 
                                           + price 
                                           + region 
                                           + Apr 
                                           + May 
                                           + Jun 
                                           + Aug 
                                           + Nov 
                                           + Dec),
                           m = 126 ,plotit = FALSE)

# 5. Calculate RMSE for training model
# mod2_fwd
trc <- trainControl(method = 'LOOCV')

mod2_fwd_RMSE <- sum( (cv_mod2_fwd[,"sales"]-cv_mod2_fwd[,"cvpred"])^2  )  / nrow(df2_trainingset)
sqrt(mod2_fwd_RMSE)

mod2_fwd_RMSEtrc <- train(sales ~ tv 
            + banner 
            + prom 
            + price 
            + region 
            + Mar 
            + Apr 
            + May 
            + Jun 
            #+ Jul 
            + Aug 
            #+ Oct 
            + Nov 
            + Dec,
            data = df2_trainingset,
            method = 'lm',
            trControl = trc)
mod2_fwd_RMSEtrc
# RMSE for mod2_fwd = 320

# mod2_stpw
trc <- trainControl(method = 'LOOCV')

mod2_stpw_RMSE <- sum( (cv_mod2_stpw[,"sales"]-cv_mod2_stpw[,"cvpred"])^2  )  / nrow(df2_trainingset)
sqrt(mod2_stpw_RMSE)

mod2_stpw_RMSEtrc <- train(sales ~ tv 
                           + banner 
                           + prom 
                           + price 
                           + region 
                           + Apr 
                           + May 
                           + Jun 
                           + Aug 
                           + Nov 
                           + Dec,
                          data = df2_trainingset,
                          method = 'lm',
                          trControl = trc)
mod2_stpw_RMSEtrc
# RMSE for mod2_stpw = 320

# 6. Create final model after cross validation
# mod2_fwd
final_mod2_fwd <- lm(sales ~ tv 
                     + banner 
                     + prom 
                     + price 
                     + region 
                     + Mar 
                     + Apr 
                     + May 
                     + Jun 
                     #+ Jul 
                     + Aug 
                     #+ Oct 
                     + Nov 
                     + Dec,
                     data = df2_trainingset)

# mod2_stpw
final_mod2_stpw <- lm(sales ~ tv 
                      + banner 
                      + prom 
                      + price 
                      + region 
                      + Apr 
                      + May 
                      + Jun 
                      + Aug 
                      + Nov 
                      + Dec,
                     data = df2_trainingset)

# 7. CV test set with training model
mod2_test_fwd <- predict.lm(object = final_mod2_fwd,newdata = df2_testset)

mod2_test_stpw <- predict.lm(object = final_mod2_stpw,newdata = df2_testset)

# Both also can use to calculate RMSE
# Calculating RMSE for test set
mod2_test_fwdRMSE = sum( (df2_testset[,"sales"]-mod2_test_fwd)^2 ) / nrow(df2_testset)
sqrt(mod2_test_fwdRMSE)
# RMSE for mod2_test_fwd = 295

mod2_test_stpwRMSE = sum( (df2_testset[,"sales"]-mod2_test_stpw)^2 ) / nrow(df2_testset)
sqrt(mod2_test_stpwRMSE)
# RMSE for mod2_test_fwd = 295

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
# 1. Split data into train and test set
set.seed(123)
n = nrow(df2)
df2_trainingsetsize <- round(n*.7)
df2_randomselection <- sample(x=1:nrow(df2), size=df2_trainingsetsize)
df2_trainingset <- df2[df2_randomselection,]
df2_testset <- df2[-df2_randomselection,]

# 2. Perform variable selection
# Forward Selection
mod3_slc1 <- ols_step_forward_p(mod3)
mod3_slc1
plot(mod3_slc3)
# RESULT:
# R2 = 0.982
# Adj R2 = 0.980
# RMSE = 291.306
# Months removed = Mar, Jul, Aug, Oct

# Backward Selection
mod3_slc2 <- ols_step_backward_p(mod3)
mod3_slc2
plot(mod3_slc3)
##### No variables have been removed, hence discard this method
# RESULT:
# R2 = 0.982
# Adj R2 = 0.980
# RMSE = 290.913
# Months removed = Aug, Dec

# Stepwise Selection
mod3_slc3 <- ols_step_both_p(mod3)
mod3_slc3
plot(mod3_slc3)
# RESULT:
# R2 = 0.982
# Adj R2 = 0.980
# RMSE = 291.306
# Months removed = Jul, Aug, Oct

# 3. Create new model after variable selection
mod3_fwd <- lm(sales ~ tv 
               + banner 
               + prom 
               + price 
               + region 
               + Jan
               + Feb
               + Apr 
               + May 
               + Jun 
               + Sep
               #+ Dec
               , data=df2)
summary(mod3_fwd)
# Summary shows Dec is not significant, hence remove it.
# Total variables remove: Mar, Jul, Aug, Oct, Dec

mod3_bwd <- lm(sales ~ tv 
               + banner 
               + prom 
               + price 
               + region 
               + Jan
               + Feb
               #+ Mar
               + Apr 
               + May 
               + Jun
               #+ Jul
               + Sep
               #+ Oct
               , data=df2)
summary(mod3_bwd)
# Summary shows Mar, Jul and Oct are not significant, hence remove it.
# Total variables remove: Aug, Dec, Mar, Jul and Oct

mod3_stpw <- lm(sales ~ tv 
               + banner 
               + prom 
               + price 
               + region 
               + Jan
               + Feb
               #+ Mar
               + Apr 
               + May 
               + Jun
               + Sep
               #+ Dec
               , data=df2)
summary(mod3_stpw)
# Summary shows Mar and Dec are not significant. Hence, remove it.
# Total variables removed: Jul, Aug, Oct, Mar and Dec

# 4. Performing CV on training model
# model3_fwd
cv_mod3_fwd <- cv.lm(data=df2_trainingset,
                     form.lm=formula(sales ~ tv 
                                     + banner 
                                     + prom 
                                     + price 
                                     + region 
                                     + Jan
                                     + Feb
                                     + Apr 
                                     + May 
                                     + Jun 
                                     + Sep),
                     m = 126 ,plotit = FALSE) 

# model3_bwd
cv_mod3_bwd <- cv.lm(data=df2_trainingset,
                      form.lm=formula(sales ~ tv 
                                      + banner 
                                      + prom 
                                      + price 
                                      + region 
                                      + Jan
                                      + Feb
                                      + Apr 
                                      + May 
                                      + Jun
                                      + Sep),
                      m = 126 ,plotit = FALSE)

# model3_stpw
cv_mod3_stpw <- cv.lm(data=df2_trainingset,
                      form.lm=formula(sales ~ tv 
                                      + banner 
                                      + prom 
                                      + price 
                                      + region 
                                      + Jan
                                      + Feb
                                      + Apr 
                                      + May 
                                      + Jun
                                      + Sep),
                      m = 126 ,plotit = FALSE)

# 5. Calculate RMSE for training model
# mod3_fwd
trc <- trainControl(method = 'LOOCV')

mod3_fwd_RMSE <- sum( (cv_mod3_fwd[,"sales"]-cv_mod3_fwd[,"cvpred"])^2  )  / nrow(df2_trainingset)
sqrt(mod3_fwd_RMSE)

mod3_fwd_RMSEtrc <- train(sales ~ tv 
                          + banner 
                          + prom 
                          + price 
                          + region 
                          + Jan
                          + Feb
                          + Apr 
                          + May 
                          + Jun 
                          + Sep,
                          data = df2_trainingset,
                          method = 'lm',
                          trControl = trc)
mod3_fwd_RMSEtrc
# RMSE for mod2_fwd = 315

# mod3_bwd
trc <- trainControl(method = 'LOOCV')

mod3_bwd_RMSE <- sum( (cv_mod3_bwd[,"sales"]-cv_mod3_bwd[,"cvpred"])^2  )  / nrow(df2_trainingset)
sqrt(mod3_bwd_RMSE)

mod3_bwd_RMSEtrc <- train(sales ~ tv 
                           + banner 
                           + prom 
                           + price 
                           + region 
                           + Jan
                           + Feb
                           + Apr 
                           + May 
                           + Jun
                           + Sep,
                           data = df2_trainingset,
                           method = 'lm',
                           trControl = trc)
mod3_bwd_RMSEtrc
# RMSE for mod2_stpw = 315

# mod3_stpw
trc <- trainControl(method = 'LOOCV')

mod3_stpw_RMSE <- sum( (cv_mod3_stpw[,"sales"]-cv_mod3_stpw[,"cvpred"])^2  )  / nrow(df2_trainingset)
sqrt(mod3_stpw_RMSE)

mod3_stpw_RMSEtrc <- train(sales ~ tv 
                           + banner 
                           + prom 
                           + price 
                           + region 
                           + Jan
                           + Feb
                           + Apr 
                           + May 
                           + Jun
                           + Sep,
                           data = df2_trainingset,
                           method = 'lm',
                           trControl = trc)
mod3_stpw_RMSEtrc
# RMSE for mod2_stpw = 315

# 6. Create final model after cross validation
# mod2_fwd
final_mod3_fwd <- lm(sales ~ tv 
                     + banner 
                     + prom 
                     + price 
                     + region 
                     + Jan
                     + Feb
                     + Apr 
                     + May 
                     + Jun
                     + Sep,
                     data = df2_trainingset)

# mod3_bwd
final_mod3_bwd <- lm(ales ~ tv 
                     + banner 
                     + prom 
                     + price 
                     + region 
                     + Jan
                     + Feb
                     + Apr 
                     + May 
                     + Jun
                     + Sep,
                      data = df2_trainingset)

# mod3_stpw
final_mod3_stpw <- lm(sales ~ tv 
                      + banner 
                      + prom 
                      + price 
                      + region 
                      + Jan
                      + Feb
                      + Apr 
                      + May 
                      + Jun
                      + Sep,
                      data = df2_trainingset)

# 7. CV test set with training model
mod3_test_fwd <- predict.lm(object = final_mod3_fwd,newdata = df2_testset)

mod3_test_bwd <- predict.lm(object = final_mod3_bwd,newdata = df2_testset)

mod3_test_stpw <- predict.lm(object = final_mod3_stpw,newdata = df2_testset)

# Both also can use to calculate RMSE
# Calculating RMSE for test set
mod3_test_fwdRMSE = sum( (df2_testset[,"sales"]-mod3_test_fwd)^2 ) / nrow(df2_testset)
sqrt(mod3_test_fwdRMSE)
# RMSE for mod2_test_fwd = 300

mod3_test_stpwRMSE = sum( (df2_testset[,"sales"]-mod3_test_bwd)^2 ) / nrow(df2_testset)
sqrt(mod3_test_stpwRMSE)
# RMSE for mod2_test_fwd = 300

mod3_test_stpwRMSE = sum( (df2_testset[,"sales"]-mod3_test_stpw)^2 ) / nrow(df2_testset)
sqrt(mod3_test_stpwRMSE)
# RMSE for mod2_test_fwd = 300

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
# 1. Split data into train and test set
set.seed(123)
n = nrow(df2)
df2_trainingsetsize <- round(n*.7)
df2_randomselection <- sample(x=1:nrow(df2), size=df2_trainingsetsize)
df2_trainingset <- df2[df2_randomselection,]
df2_testset <- df2[-df2_randomselection,]

# 2. Perform variable selection
# Forward Selection
mod3_slc1 <- ols_step_forward_p(mod3)
mod3_slc1
plot(mod3_slc3)
# RESULT:
# R2 = 0.982
# Adj R2 = 0.980
# RMSE = 291.306
# Months removed = Mar, Jul, Aug, Oct

# Backward Selection
mod3_slc2 <- ols_step_backward_p(mod3)
mod3_slc2
plot(mod3_slc3)
##### No variables have been removed, hence discard this method
# RESULT:
# R2 = 0.982
# Adj R2 = 0.980
# RMSE = 290.913
# Months removed = Aug, Dec

# Stepwise Selection
mod4_slc3 <- ols_step_both_p(mod4)
mod4_slc3
plot(mod4_slc3)
# RESULT:
# R2 = 0.982
# Adj R2 = 0.980
# RMSE = 291.306
# Months removed = Jul, Aug, Oct

# 3. Create new model after variable selection
mod3_fwd <- lm(sales ~ tv 
               + banner 
               + prom 
               + price 
               + region 
               + Jan
               + Feb
               + Apr 
               + May 
               + Jun 
               + Sep
               #+ Dec
               , data=df2)
summary(mod3_fwd)
# Summary shows Dec is not significant, hence remove it.
# Total variables remove: Mar, Jul, Aug, Oct, Dec

mod3_bwd <- lm(sales ~ tv 
               + banner 
               + prom 
               + price 
               + region 
               + Jan
               + Feb
               #+ Mar
               + Apr 
               + May 
               + Jun
               #+ Jul
               + Sep
               #+ Oct
               , data=df2)
summary(mod3_bwd)
# Summary shows Mar, Jul and Oct are not significant, hence remove it.
# Total variables remove: Aug, Dec, Mar, Jul and Oct

mod4_stpw <- lm(sales ~ tv 
                + banner 
                + prom 
                + price 
                + region 
                + Jan
                + Feb
                + Apr 
                + May 
                + Jun
                + Sep
                , data=df2)
summary(mod4_stpw)

# 4. Performing CV on training model
# model3_fwd
cv_mod3_fwd <- cv.lm(data=df2_trainingset,
                     form.lm=formula(sales ~ tv 
                                     + banner 
                                     + prom 
                                     + price 
                                     + region 
                                     + Jan
                                     + Feb
                                     + Apr 
                                     + May 
                                     + Jun 
                                     + Sep),
                     m = 126 ,plotit = FALSE) 

# model3_bwd
cv_mod3_bwd <- cv.lm(data=df2_trainingset,
                     form.lm=formula(sales ~ tv 
                                     + banner 
                                     + prom 
                                     + price 
                                     + region 
                                     + Jan
                                     + Feb
                                     + Apr 
                                     + May 
                                     + Jun
                                     + Sep),
                     m = 126 ,plotit = FALSE)

# model3_stpw
cv_mod4_stpw <- cv.lm(data=df2_trainingset,
                      form.lm=formula(sales ~ tv 
                                      + banner 
                                      + prom 
                                      + price 
                                      + region 
                                      + Jan
                                      + Feb
                                      + Apr 
                                      + May 
                                      + Jun
                                      + Sep),
                      m = 126 ,plotit = FALSE)

# 5. Calculate RMSE for training model
# mod3_fwd
trc <- trainControl(method = 'LOOCV')

mod3_fwd_RMSE <- sum( (cv_mod3_fwd[,"sales"]-cv_mod3_fwd[,"cvpred"])^2  )  / nrow(df2_trainingset)
sqrt(mod3_fwd_RMSE)

mod3_fwd_RMSEtrc <- train(sales ~ tv 
                          + banner 
                          + prom 
                          + price 
                          + region 
                          + Jan
                          + Feb
                          + Apr 
                          + May 
                          + Jun 
                          + Sep,
                          data = df2_trainingset,
                          method = 'lm',
                          trControl = trc)
mod3_fwd_RMSEtrc
# RMSE for mod2_fwd = 320

# mod3_bwd
trc <- trainControl(method = 'LOOCV')

mod3_bwd_RMSE <- sum( (cv_mod3_bwd[,"sales"]-cv_mod3_bwd[,"cvpred"])^2  )  / nrow(df2_trainingset)
sqrt(mod3_bwd_RMSE)

mod3_bwd_RMSEtrc <- train(sales ~ tv 
                          + banner 
                          + prom 
                          + price 
                          + region 
                          + Jan
                          + Feb
                          + Apr 
                          + May 
                          + Jun
                          + Sep,
                          data = df2_trainingset,
                          method = 'lm',
                          trControl = trc)
mod3_bwd_RMSEtrc
# RMSE for mod2_stpw = 320

# mod4_stpw
trc <- trainControl(method = 'LOOCV')

mod4_stpw_RMSE <- sum( (cv_mod4_stpw[,"sales"]-cv_mod4_stpw[,"cvpred"])^2  )  / nrow(df2_trainingset)
sqrt(mod4_stpw_RMSE)

train_prediction2 <- predict(final_mod3_stpw, newdata = df2_trainingset)

MSE_train2 <- sum((df2_trainingset[,'sales'] - train_prediction2)^2 / nrow(df2_trainingset))
RMSE_train2 <- sqrt(MSE_train2)
RMSE_train2

CVMSPE2<- sum( (cv_mod4_stpw[,"sales"]-cv_mod4_stpw[,"cvpred"])^2  )  / nrow(df2_trainingset)
CVMSPE2
sqrt(CVMSPE2)

mod4_stpw_RMSEtrc <- train(sales ~ tv 
                           + banner 
                           + prom 
                           + price 
                           + region 
                           + Jan
                           + Feb
                           + Apr 
                           + May 
                           + Jun
                           + Sep,
                           data = df2_trainingset,
                           method = 'lm',
                           trControl = trc)
mod3_stpw_RMSEtrc
# RMSE for mod2_stpw = 

# 6. Create final model after cross validation
### FINAL MODEL FROM KEM ###
final_model <- lm(sales ~ tv + banner + prom 
                  + price 
                  + region 
                  # + Dec
                  + Apr
                  + May
                  + Jun
                  + Sep
                  + Jan
                  + Feb,
                  data = train)

summary(final_model)


# mod2_fwd
final_mod3_fwd <- lm(sales ~ tv 
                     + banner 
                     + prom 
                     + price 
                     + region 
                     + Jan
                     + Feb
                     + Apr 
                     + May 
                     + Jun
                     + Sep,
                     data = df2_trainingset)

# mod3_bwd
final_mod3_bwd <- lm(ales ~ tv 
                     + banner 
                     + prom 
                     + price 
                     + region 
                     + Jan
                     + Feb
                     + Apr 
                     + May 
                     + Jun
                     + Sep,
                     data = df2_trainingset)

# mod3_stpw
final_mod3_stpw <- lm(sales ~ tv 
                      + banner 
                      + prom 
                      + price 
                      + region 
                      + Jan
                      + Feb
                      + Apr 
                      + May 
                      + Jun
                      + Sep,
                      data = df2_trainingset)

# 7. CV test set with training model
mod3_test_fwd <- predict.lm(object = final_mod3_fwd,newdata = df2_testset)

mod3_test_bwd <- predict.lm(object = final_mod3_bwd,newdata = df2_testset)

mod3_test_stpw <- predict.lm(object = final_mod3_stpw,newdata = df2_testset)

# Both also can use to calculate RMSE
# Calculating RMSE for test set
mod3_test_fwdRMSE = sum( (df2_testset[,"sales"]-mod3_test_fwd)^2 ) / nrow(df2_testset)
sqrt(mod3_test_fwdRMSE)
# RMSE for mod2_test_fwd = 295

mod3_test_stpwRMSE = sum( (df2_testset[,"sales"]-mod3_test_bwd)^2 ) / nrow(df2_testset)
sqrt(mod3_test_stpwRMSE)
# RMSE for mod2_test_fwd = 295

mod3_test_stpwRMSE = sum( (df2_testset[,"sales"]-mod3_test_stpw)^2 ) / nrow(df2_testset)
sqrt(mod3_test_stpwRMSE)
# RMSE for mod2_test_fwd = 295

train_prediction2 <- predict(final_mod3_stpw, newdata = df2_trainingset)

MSE_train2 <- sum((df2_trainingset[,'sales'] - train_prediction2)^2 / nrow(df2_trainingset))
RMSE_train2 <- sqrt(MSE_train2)
RMSE_train2










































model2 <- lm(sales ~ tv + banner + prom 
             + price 
             + region 
             # + Dec
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
             + Nov, 
             data = df2)
summary(model2)
# R-sq in model 1 and 2 is identical 
summary(model1) # R-sq = 0.982
summary(model2) # 0.982
# RMSE = 292.5256

x3 <- ols_step_both_p(model2)
plot(x3)
x4 <- ols_step_both_aic(model2)

stepwise <- lm(sales ~ tv + banner + prom 
               + price 
               + region 
               # + Dec
               + Apr
               + May
               + Jun
               + Sep
               + Jan
               + Feb, 
               data = df2)

summary(stepwise) # R-sq = 0.981 

require(caret)
require(caTools)

set.seed(123)

data_split = sample.split(df2$sales, # split on any columns in dataframe
                          SplitRatio = .7) # 70 percent training

train = subset(df2, data_split == TRUE) # return 70% if true, else 30 if FALSE
test = subset(df2, data_split == FALSE)

# Cross validation (LOOCV)
require(DAAG)
train_ctrl <- trainControl(method = 'LOOCV')

require(randomForest)

LCV_1 = train(sales ~ tv + banner + prom 
              + price 
              + region 
              # + Dec
              + Apr
              + May
              + Jun
              + Sep
              + Jan
              + Feb,
              data = train,
              method = 'lm',
              trControl = train_ctrl)

LCV_1 # RMSE = 301 
#####

LCV_result <- cv.lm(data = train,
                    form.lm = formula(sales ~ tv + banner + prom 
                                      + price 
                                      + region 
                                      # + Dec
                                      + Apr
                                      + May
                                      + Jun
                                      + Sep
                                      + Jan
                                      + Feb),
                    m = nrow(train),
                    plotit = FALSE)

# Calculate MSE
MSE <- sum((LCV_result[, 'sales'] - LCV_result[, 'cvpred'])^2/nrow(train))

# Calculate RMSE
RMSE <- sqrt(MSE)
RMSE # = 301 error on average 

# Predict on training set and test set
  # Compare RMSE

final_model <- lm(sales ~ tv + banner + prom 
                  + price 
                  + region 
                  # + Dec
                  + Apr
                  + May
                  + Jun
                  + Sep
                  + Jan
                  + Feb,
                  data = train)

summary(final_model)

# Train error
train_prediction <- predict(final_model, newdata = train)

MSE_train <- sum((train[,'sales'] - train_prediction)^2 / nrow(train))
RMSE_train <- sqrt(MSE_train)
RMSE_train # Fitted model returns 266 error on average on training data 

# Test error
test_prediction <- predict(final_model, newdata = test)

MSE_test <- sum((test[,'sales'] - test_prediction)^2 / nrow(test))
RMSE_test <- sqrt(MSE_test)
RMSE_test # Fitted model returns 346 error on average on training data 
# Our fitted model's sales predictions deviates from observed sales by an average of 346 packs

# Train model residual plot 
require(Rmisc)
train$prediction <- predict(final_model, newdata = train)
train$residuals <- train$sales - train$prediction

a <- ggplot(train, aes(x = prediction,
                     y = sales)) +
  geom_point(color = 'orchid') + 
  geom_abline(color = 'orange') + 
  ggtitle('Train result 1') + 
  theme_classic()


b <- ggplot(data = train,
       aes(x = prediction,
           y = residuals)) + 
  geom_pointrange(aes(ymin = 0,
                      ymax = residuals),
                  color = 'orchid') +
  geom_hline(yintercept = 0,
             linetype = 1,
             color = 'orange') + 
  ggtitle('Train result 2') + 
  theme_classic()

multiplot(a, b)

test$prediction <- predict(final_model, newdata = test)
test$residuals <- test$sales - test$prediction

c <- ggplot(test, aes(x = prediction,
                       y = sales)) +
  geom_point(color = 'tomato1') + 
  geom_abline(color = 'seagreen') + 
  ggtitle('Test result 1') + 
  theme_classic()


d <- ggplot(data = test,
            aes(x = prediction,
                y = residuals)) + 
  geom_pointrange(aes(ymin = 0,
                      ymax = residuals),
                  color = 'tomato1') + 
  geom_hline(yintercept = 0,
             linetype = 1,
             color = 'seagreen') + 
  ggtitle('Test result 2') + 
  theme_classic()

multiplot(c, d)

# Because the test set is smaller than training, it is expected that the test RMSE is higher than the train RMSE
# Our fitted model's test and train RMSE is not significantly different from each other

model2 <- lm(sales ~ tv + banner + prom + month, data = df)
summary(model2)
sales_tv <- lm(sales~tv, data=df)
summary(sales_tv)
sales_banner <- lm(sales~banner, data=df)
summary(sales_banner)
sales_prom <- lm(sales~prom, data=df)
summary(sales_prom)
plot(sales_tv)

########## TEST CODE ##########
mkt1 <- lm(sales~ tv, data = df)
mkt2 <- lm(sales~ banner, data = df)
mkt3 <- lm(sales~ prom, data = df)
summary(mkt1)
summary(mkt2)
summary(mkt3)

cor(df$tv, df$sales)
plot(df$tv, df$sales, xlab = "TV", ylab = 'Sales')
abline(lm(sales ~ tv, data = df))
cor(df$banner, df$sales)
plot(df$banner, df$sales, xlab = "Banner", ylab = 'Sales')
abline(lm(sales ~ banner, data = df))
cor(df$prom, df$sales)
plot(df$prom, df$sales, xlab = "Prom", ylab = 'Sales')
abline(lm(sales ~ prom, data = df))


df$mkt <- ifelse(df$tv | df$banner | df$prom, 1,0)
t.test(df[df$mkt == 0, 2][[1]], df[df$mkt !=0, 2][[1]])


ggplot(df, aes(mkt, sales)) +
  geom_boxplot(stat = 'summary', 
           fun.y = 'mean',
           fill = 'yellow') + 
  geom_label(stat = 'count',
             aes(label = ..count..,
                 y = ..count..),
             size = 5) + 
  theme_light() + 
  geom_hline(yintercept = 4094,
             linetype = 'dashed',
             color ='firebrick3') + 
  labs(title = 'Sales with no marketing activity vs Sales with marketing activity')

df$region <- as.factor(df$region)
df$region <- as.numeric(df$region)
model4 <- lm(sales ~ region, data=df)
summary(model4)
df$region <- as.factor(df$region)
str(df)


df$North <- as.numeric(df$region = 'The North')
df$South <- as.numeric(df$region = 'The South')
df$West <- as.numeric(df$region = 'West')
df$East <- as.numeric(df$region = 'East')
df$Capital <- as.numeric(df$region = 'Capital')

mean(df$sales)


str(qkitch_region)
model3 <- lm(sales ~region, data=qkitch_region)
summary(model3)
qkitch_region$region <- as.factor(qkitch_region$region)
model4 <- lm(sales ~region, data=qkitch_region)
summary(model4)
str(qkitch_region)

require(caret)
trc <- trainControl(method = 'LOOCV')

cv1 <- train(income ~ store_production + downtown_store,
             data = trainingset,
             method = 'lm',
             trControl = trc)

cv1 # 0.318s

########## Essential Codes ##########
# Doing forward variable selection
selection <-ols_step_forward_p(model1) 

# Plotting R-square and RMSE (multi-plot) to see variable selection effect
stat <- read.csv('object1.csv')
x1 <- ggplot(data = stat, aes(x = Step,
                              y = `R-square`)) + 
  geom_point(color = 'deepskyblue') + 
  geom_line(color = 'deeppink3') + 
  annotate('text', 
           x = 1.4, y = 0.564,
           label = 'Region',
           color = 'black') + 
  annotate('text', 
           x = 2.3, y = 0.761,
           label = 'April',
           color = 'black') + 
  annotate('text', 
           x = 3.6, y = 0.909,
           label = 'Promotions',
           color = 'black') + 
  annotate('text', 
           x = 4.5, y = 0.934,
           label = 'TV ads',
           color = 'black') + 
  annotate('text', 
           x = 5.4, y = 0.951,
           label = 'Price',
           color = 'black') + 
  annotate('text', 
           x = 6.3, y = 0.964,
           label = 'May',
           color = 'black') + 
  annotate('text', 
           x = 7, y = 0.972,
           label = 'Banner ads',
           color = 'black') + 
  annotate('text', 
           x = 8.2, y = 0.973,
           label = 'August',
           color = 'black') + 
  annotate('text', 
           x = 9.4, y = 0.975,
           label = 'February',
           color = 'black') +
  annotate('text', 
           x = 10.2, y = 0.976,
           label = 'June',
           color = 'black') + 
  annotate('text', 
           x = 11.4, y = 0.977,
           label = 'November',
           color = 'black') + 
  ggtitle('Variable selection process (step selection by p-value)') + 
  theme_light()



x2 <- ggplot(data = step, aes(x = Step,
                              y = RMSE)) + 
  geom_point(color = 'deepskyblue') + 
  geom_line(color = 'deeppink3') + 
  annotate('text', 
           x = 1.4, y = 1410.4773,
           label = 'Region',
           color = 'black') + 
  annotate('text', 
           x = 2.3, y = 1045.8945,
           label = 'April',
           color = 'black') + 
  annotate('text', 
           x = 3.6, y = 649.3087,
           label = 'Promotions',
           color = 'black') + 
  annotate('text', 
           x = 4.5, y = 554.3976,
           label = 'TV ads',
           color = 'black') + 
  annotate('text', 
           x = 5.4, y = 475.9810,
           label = 'Price',
           color = 'black') + 
  annotate('text', 
           x = 6.3, y = 409.1163,
           label = 'May',
           color = 'black') + 
  annotate('text', 
           x = 7, y = 364.1601,
           label = 'Banner ads',
           color = 'black') + 
  annotate('text', 
           x = 8.2, y = 356.6843,
           label = 'August',
           color = 'black') + 
  annotate('text', 
           x = 9.4, y = 345.4055,
           label = 'February',
           color = 'black') +
  annotate('text', 
           x = 10.2, y = 340.1393,
           label = 'June',
           color = 'black') + 
  annotate('text', 
           x = 11.4, y = 335.0036,
           label = 'November',
           color = 'black') + 
  theme_light()

multiplot(x1, x2)

# Performing CV on training model
cvresult <- cv.lm(data=trainingset,form.lm =formula(income ~ store_production + downtown_store),m = 64 ,plotit = FALSE) 

# Create final optimal model using the cross-validated training model
final_mod<- lm(income ~ store_production + downtown_store,data = trainingset) 

# Find predicted value using the final optimial model (Means use train model to apply on test dataset)
predicted_values<- predict.lm(object = final_mod,newdata = testset)

# Both also can use to calculate RMSe
# Calculating RMSE for test set
testsetMSPE=sum( (testset[,"income"]-predicted_values)^2 ) / nrow(testset)
testsetMSPE
sqrt(testsetMSPE)

##### OR

# Calculating RMSE for training model
trc <- trainControl(method = 'LOOCV')

cv1 <- train(sales ~ region
             + Apr
             + promotions
             + Tv 
             + price
             + May
             + banner
             + Aug
             + Feb
             + Jun
             + Nov,
             data = train,
             method = 'lm',
             trControl = trc)
cv1 

# After getting RMSE for train and test, evaluate it
# Can keep on adding or remove variable to get lowest RMSE to find the optimal model
# CANNOT REMOVE VARIABLE BASED ON RMSE OF TEST DATASET BECAUSE IT'S CHEATING
# IF USE TEST DATASET TO DECIDE WHICH VARIABLE TO REMOVE IT BECOMES THE TRAIN MODEL ALREADY

# Create dummy variables for months
df$Jan <- as.numeric(df$month == 'Jan')
df$Feb <- as.numeric(df$month == 'Feb')
df$Mar <- as.numeric(df$month == 'Mar')
df$Apr <- as.numeric(df$month == 'Apr')
df$May <- as.numeric(df$month == 'May')
df$Jun <- as.numeric(df$month == 'Jun')
df$Jul <- as.numeric(df$month == 'Jul')
df$Aug <- as.numeric(df$month == 'Aug')
df$Sep <- as.numeric(df$month == 'Sep')
df$Oct <- as.numeric(df$month == 'Oct')
df$Nov <- as.numeric(df$month == 'Nov')
df$Dec <- as.numeric(df$month == 'Dec')

month <- c('Jan', 'Feb', 'Mar',
           'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep',
           'Oct', 'Nov', 'Dec')

df[month] <- lapply(df[month], factor)
df = subset(df, select = -c(product))

# Anova test to see difference/variance in sales between every month in Month variable (and other non-mkt var)
# H0 = no difference
# H1 = difference
summary(aov(sales ~ month, data = df))
summary(aov(sales ~ region, data = df))
summary(aov(sales ~ prom, data = df))
summary(aov(sales ~ month, data=df))
boxplot(sales ~ month, data=df, main="Monthly Sales", ylab='Sales', xlab='Month')

# Plotting for RMSE and variable selection
ggplot(data = step, aes(x = Step,
                        y = RMSE)) + 
  geom_point(color = 'deepskyblue') + 
  geom_line(color = 'deeppink3') + 
  annotate('text', 
           x = 1.4, y = 1410,
           label = 'Region',
           color = 'black') + 
  annotate('text', 
           x = 2.3, y = 1046,
           label = 'April',
           color = 'black') + 
  annotate('text', 
           x = 3.6, y = 649,
           label = 'Promotions',
           color = 'black') + 
  annotate('text', 
           x = 4.5, y = 554,
           label = 'TV ads',
           color = 'black') + 
  annotate('text', 
           x = 5.4, y = 476,
           label = 'Price',
           color = 'black') + 
  annotate('text', 
           x = 6.3, y = 409,
           label = 'May',
           color = 'black') + 
  annotate('text', 
           x = 7, y = 350,
           label = 'Banner ads',
           color = 'black') + 
  annotate('text', 
           x = 8.2, y = 340,
           label = 'August',
           color = 'black') + 
  annotate('text', 
           x = 9.4, y = 330,
           label = 'February',
           color = 'black') +
  annotate('text', 
           x = 10.2, y = 335,
           label = 'June',
           color = 'black') + 
  annotate('text', 
           x = 11.4, y = 325,
           label = 'November',
           color = 'black') + 
  ggtitle('Variable selection process') + 
  theme_light()


# T-test for marketing variables and sales (test presence and absence effect)
t.test(df[df$marketing == 0, 2][[1]], df[df$marketing !=0, 2][[1]])

# T-test plot for marketing variables and sales (test presence and absence effect)
ggplot(df, aes(Marketing, sales)) +
  geom_bar(stat = 'summary', 
           fun.y = 'median',
           fill = 'seagreen') + 
  geom_label(stat = 'count',
             aes(label = ..count..,
                 y = ..count..),
             size = 5) + 
  theme_light() + 
  geom_hline(yintercept = 5162.478,
             linetype = 'dashed',
             color ='firebrick3') + 
  labs(title = 'Sales with no marketing activity vs Sales with marketing activity')

# Plotting Cross Validation of Training Model
x1 <- ggplot(cvresult, aes(x = cvpred,
                           y = sales)) + 
  geom_point(color = 'mediumorchid3',
             alpha = 0.8) + 
  geom_abline(color = 'green') + 
  ggtitle('Cross validation sales prediction (training data)')

cvresult$residuals <- cvresult$sales - cvresult$cvpred

x2 <- ggplot(data = cvresult,
             aes(x = cvpred,
                 y = residuals)) + 
  geom_pointrange(aes(ymin = 0,
                      ymax = residuals),
                  color = 'purple',
                  alpha = 0.8) + 
  geom_hline(yintercept = 0,
             linetype = 1,
             color = 'black') + 
  ggtitle('Cross validation residuals & predicted sales (training data)')

require(Rmisc)
multiplot(x1, x2)

# Plotting RMSE of test data using training model
x1 <- ggplot(test, aes(x = predictions,
                       y = sales)) +
  geom_point(color = 'deepskyblue',
             alpha = 0.8) + 
  geom_abline(color = 'firebrick') + 
  ggtitle('Actual & Predicted (test data)')

test$residuals <- test$sales - test$predictions

x2 <- ggplot(data = test,
             aes(x = predictions,
                 y = residuals)) + 
  geom_pointrange(aes(ymin = 0,
                      ymax = residuals),
                  color = 'black',
                  alpha = 0.8) + 
  geom_hline(yintercept = 0,
             linetype = 1,
             color = 'firebrick') + 
  ggtitle('Residuals & Predicted (test data)')

plot(trainingset$store_production,trainingset$income)
#wait
points(x = trainingset$store_production, y = fitted_mod$fitted.values,pch=18,col=rgb(0.5,0,0,.8),type="p")

# 4) Let's compute the residuals of the model.
residuals<- trainingset$income - predict.lm(object = fitted_mod,newdata = trainingset) #Remember that the predict function give us the predicted values from the model.

# 5) Plots of the residuals
# Index vs. residuals
plot(1:length(residuals), residuals,bty="n",main=NULL,pch=18,col=rgb(0.6,.7,.2,.9),cex.axis=2,cex=1,cex.main=1,cex.lab=3,xlab="Index",ylab="Residuals")
abline(h=0,col="red") #Creates a horizontal red line at y=0

#Fitted values vs. residuals
plot(fitted_mod$fitted.values, residuals,bty="n",main=NULL,pch=18,col=rgb(0.2,.7,.7,.9),cex.axis=2,cex=1,cex.main=3,cex.lab=3,xlab="Fitted values",ylab="residuals")
abline(h=0,col="red")

#distance_to_towncentre vs. residuals
plot(trainingset$distance_to_towncentre, residuals,bty="n",main=NULL,pch=18,col=rgb(0.3,.7,.2,.9),cex.axis=2,cex=1,cex.main=3,cex.lab=3,xlab="Location",ylab="residuals")
abline(h=0,col="red")

#QQplot for the residuals.
qqnorm(y = fitted_mod$residuals,cex.axis=2,cex=1,cex.main=3,cex.lab=3) #Creates the qqplot for the residuals by comparint the quantiles of the residuals to the quantiles of a standard normal distribution. The expected plot should show the points in a line.
qqline(fitted_mod$residuals, col = "red", lwd = 2) #Adds the line in red where the points are expected to be located.

#Histogram of the residuals
hist(residuals,bty="n",main=NULL,pch=18,col=rgb(0.2,.7,.7,.9),cex.axis=2,cex=1,cex.main=3,cex.lab=3,freq = FALSE)
points(x = seq(min(residuals),max(residuals),length.out = 100), dnorm(x = seq(min(residuals),max(residuals),length.out = 100),mean = 0, sd = sd(residuals)), col="red",type="l")

########## NOTES:
# 1. Use as.factor when it is numerical. If it's character you can straightaway insert in your lm model
# 2. rho != 0 means not Null Hypothesis
# 3. If p-value is insignificant (>0.05), then the month is no difference with the left-out month
# 4. Build residual histogram -> refer lab 8

