# Version: Model 2 (All Months)
#Import all library
library(readxl)
require(ggplot)
require(randomForest)
require(DAAG)
require(lmtest)
require(caret)
require(olsrr)
require(car)
require(grid)
require(gridExtra)
require(ggfortify)
require(Rmisc)
require(Metrics)
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
View(df2)
########## Create Model 2 (All months) ##########
### Initial Model ###
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
print(summary(mod2),digits=4)

### Proposed Model (Used this to estimate sales for mkt activities) ###
final_mod2 <- lm(sales ~ region 
                      + Apr 
                      + prom 
                      + tv
                      + price
                      + banner
                      + May
                      + Jun
                      + Dec
                      + Nov
                      + Aug,
                      data = df2)
print(summary(final_mod2),digits=4)

# Interpretation
# 1. F-statistic p-value < 0.05, highly significant. 
# At least one predictor variables is significantly related to the sales.
# 2. R2 = 0.982 means 98.20% of the variance in sales can be predicted by 
# all the marketing and non-marketing variables in the model.
# Calculate error rate
sigma(final_mod2)/mean(df2$sales)
# 3. RSE of 292 corresponding to 7.18% of error rate.
# 4. t-statistic p-value for tv, banner and prom < 0.05, hence they have significant association with sales
# 5. For every increase of 1 GRP in TV, 10.748 packs will be sold
# 6. For every increase of 1 GRP in Banner, 24.514 packs will be sold
# 7. For every increase of 1% of stores with Promotion, 30.571 packs will be sold
# 8. When compare other regions to Capital, it shows that Capital has relatively higher sales 
# 9. When compare other months to Capital, it shows that April has relatively higher sales

# Find correlation between sales and other variables
cor(df2[,2:6], use = 'pairwise.complete.obs', method='pearson')

### Finding association between sales & marketing variables(TV, Banner & Prom)
tv_sales <- cor(df2$tv, df2$sales, method='pearson')
tv_sales
# Correlation between sales and tv is 0.171, indicate a weak positive correlation/association.

banner_sales <- cor(df2$banner, df2$sales, method='pearson')
banner_sales
# Correlation between sales and banner is 0.020, indicate a very weak positive correlation/association.

prom_sales <- cor(df2$prom, df2$sales, method='pearson')
prom_sales
# Correlation between sales and prom is 0.547, indicate a moderate positive orrelation/association.
# Hence, all marketing variables has positive association with sales.
tv_sales
banner_sales
prom_sales
# Calculate total predicted packs sold based on Model 1
sum(predict(final_mod2)) # 736992

# How many packs sold associated to advertisement?
# 1. TV
tv_sales <- final_mod2$coefficients[8]*sum(df2$tv)
# 6390 * 10.8943 = 69615
print(tv_sales)
# Total packs sold associated with 6390 GRP of TV = 69615

# 2. Banner
banner_sales <- final_mod2$coefficient[10]*sum(df2$banner)
# 1465 * 22.3082 = 32682
print(banner_sales)
# Total packs sold associated with 1465 GRP of banner = 32682

# 3. Prom
prom_sales <- final_mod2$coefficient[7]*sum(df2$prom)
# 3945 * 30.8054 = 121527
print(prom_sales)
# Total packs sold associated with 3945 percentage of stores = 121527

# Calculate the total sales contributed by advertisement and promotion
sum(tv_sales, banner_sales)
# ads = 102296
# prom = 121527
sum(tv_sales, banner_sales, prom_sales)

# Calculate the sales that are not contributed by advertisement and promotion
sum(df2$sales) - tv_sales - banner_sales - prom_sales
# Sales contributed by non-marketing variable = 513169

# Calculate the efficiency of TV and Banner.
# TV
2000000/tv_sales
# The company need to spend £28.73 in order to get 1 pack sold

# Banner
500000/banner_sales
# The company need to invest £15.30 in order to get 1 pack sold
# Hence, banner is more efficient than tv

########## Finding other possible source of the variation in sales other than marketing variables
# Conduct Anova test to see difference/variance in sales between every month in Region and Month
# H0 = no difference
# H1 = difference

# Region
summary(aov(sales ~ region, data = df2)) # p < 0.05, shows difference in sales across regions
# Plot variance of monthly sales
ggplot(df2, aes(x=as.factor(df2$region), y=sales)) + 
  geom_boxplot(fill="green") + 
  xlab("Region") + ylab('Sales') + theme_minimal() + ggtitle("Sales Across Regions") +
  theme(plot.title = element_text(hjust = 0.5))

# Month
summary(aov(sales ~ month, data = df2)) # p < 0.05, shows difference in sales across months
# Plot variance of monthly sales
ggplot(df2, aes(x=as.factor(df2$month), y=sales)) + 
  geom_boxplot(fill="purple") + 
  xlab("Month") + ylab('Sales') + theme_minimal() + ggtitle("Monthly Sales") +
  theme(plot.title = element_text(hjust = 0.5))

# T-test for marketing variables and sales (test presence and absence effect)
# First create new col to count if marketing activities are presence
df2$TV <- ifelse(df2$tv, 1, 0)
df2$TV <- as.factor(df2$TV)

df2$Banner <- ifelse(df2$banner, 1, 0)
df2$Banner <- as.factor(df2$Banner)

df2$Prom <- ifelse(df2$prom, 1, 0)
df2$Prom <- as.factor(df2$Prom)

df2$Marketing <- ifelse(df2$tv | df2$banner | df2$prom, 1, 0 )

# Conduct t-test
t.test(df2$Marketing == 0, df2$Marketing != 0)
#t.test p-value < 0.05, hence there's difference in sales mean with and without marketing

# Conduct Anova test
summary(aov(df2$Marketing ==0 ~ df2$Marketing != 0, data=df2))
#Anova test p-value < 0.05, hence there's difference in sales variance with and without marketing

########## Perform Assumptions Checking ##########
# 1. Check Homoscedasticity
# Perform Statistical test to check the presence of heteroscedasticity
# H0 = variance of the residual is constant (homoscedasticity)
# H1 = variance of the residual is not constant (heterscedasticity exists)

lmtest::bptest(final_mod2) 
#bp test shows p-value = 0.06493 > 0.05, fail to reject H0, not enough evidence to claim non-constant variance of our residual errors
car::ncvTest(final_mod2)
#ncv test shows p-value = 0.49088 > 0.05, fail to reject H0, not enough evidence to claim non-constant variance of our residual errors

# 2. Check normality of residuals
# H0 = residual errors distribution is not statistically different from a normal distribution (normally distributed)
# H1 = residual errors distribution are statistically different from a normal distribution (not normally distributed)
ols_test_normality(final_mod2) 
# Shapiro-Wilk returns a p-value of 0.1668 > 0.05
# Kolmogorov-Smirnov test returns a p-value of 0.9639 > 0.05
# Anderson-Darling returns a p-value of 0.5572 > 0.05
# Hence, fail to reject H0. Residual is not statistically different from a normal distribution (Residual is normally distributed)

# 3. Testing the independence assumption (Autocorrelation)
# H0 = independent variables are not positively correlated with each other (No Autocorrelation)
# H1 = positive correlation exists between independent variables (Autocorrelation Exists)
dwtest(final_mod2)
durbinWatsonTest(final_mod2)

df3 <- df2[sample(nrow(df2)),]
df3

mew <- lm(sales ~ region 
                 + Apr 
                 + prom 
                 + tv
                 + price
                 + banner
                 + May
                 + Jun
                 + Dec
                 + Nov
                 + Aug,
                 data = df3)
summary(mew)

dwtest(mew)
durbinWatsonTest(mew) 
# p-value = 0.009 < 0.05, reject H0, can claim autocorrelation exists 

# 4. Testing the Multicollinearity Assumption (Variance Inflation Factor)
vif(final_mod2)
# region = 4, moderately correlated with other variables. 

########## Cross Validation Starts Here ##########
# 1. Split data into train and test set based on 80:20

# 2. Perform variable selection
# Forward
mod2fwd <- ols_step_forward_p(mod2)
mod2fwd
plot(mod2fwd)
# Variable removed: Jan, Feb, Sep
# R2: 0.982
# Adj R2: 0.980
# RMSE: 290.913

# Stepwise
mod2stpw <- ols_step_both_p(mod2)
mod2stpw
plot(mod2stpw)
# Variable removed: Jan, Feb, Mar, Jul, Sep, Oct
# R2 = 0.981
# Adj R2 = 0.980
# RMSE = 291.967

# Both forward and stepwise shows different removal of variable. Hence, CV both.

# 3. Create new model after variable selection
new_mod2fwd <- lm(sales ~ region 
                  + Apr 
                  + prom 
                  + tv
                  + price
                  + banner
                  + May
                  + Jun
                  + Dec
                  + Nov
                  + Aug
                  + Mar, 
                  data=df2)
summary(new_mod2fwd)
# Summary shows Jul, Oct are insignificant. Hence, remove it.

new_mod2stpw <- lm(sales ~ region 
                   + Apr 
                   + prom 
                   + tv
                   + price
                   + banner
                   + May
                   + Jun
                   + Dec
                   + Nov
                   + Aug, 
                   data=df2)
summary(new_mod2stpw)
# No insignificant variable, hence retain this model.


# 4. Performing CV on training model
cv_mod2fwd <- cv.lm(data=df2_trainingset,
                 form.lm=formula(sales ~ region 
                                 + Apr 
                                 + prom 
                                 + tv
                                 + price
                                 + banner
                                 + May
                                 + Jun
                                 + Dec
                                 + Nov
                                 + Aug
                                 + Mar),
                 m = 126 ,plotit = FALSE) 

cv_mod2stpw <- cv.lm(data=df2_trainingset,
                    form.lm=formula(sales ~ region 
                                    + Apr 
                                    + prom 
                                    + tv
                                    + price
                                    + banner
                                    + May
                                    + Jun
                                    + Dec
                                    + Nov
                                    + Aug),
                    m = 126 ,plotit = FALSE) 

# 5. Calculate RMSE for training model
mod2fwd_trainMSE <- sum( (cv_mod2fwd[,"sales"]-cv_mod2fwd[,"cvpred"])^2  )  / nrow(df2_trainingset)
mod2fwd_trainMSE
mod2fwd_trainRMSE <- sqrt(mod2fwd_trainMSE)
mod2fwd_trainRMSE

trc <- trainControl(method = 'LOOCV')
mod2fwd_trainRMSEtrc <- train(sales ~ region 
                              + Apr 
                              + prom 
                              + tv
                              + price
                              + banner
                              + May
                              + Jun
                              + Dec
                              + Nov
                              + Aug
                              + Mar,
                           data = df2_trainingset,
                           method = 'lm',
                           trControl = trc)
mod2fwd_trainRMSEtrc 

# RMSE for mod2fwd_trainRMSE = 315

mod2stpw_trainMSE <- sum( (cv_mod2stpw[,"sales"]-cv_mod2stpw[,"cvpred"])^2  )  / nrow(df2_trainingset)
mod2stpw_trainMSE
mod2stpw_trainRMSE <- sqrt(mod2stpw_trainMSE)
mod2stpw_trainRMSE

trc <- trainControl(method = 'LOOCV')
mod2stpw_trainRMSEtrc <- train(sales ~ region 
                              + Apr 
                              + prom 
                              + tv
                              + price
                              + banner
                              + May
                              + Jun
                              + Dec
                              + Nov
                              + Aug,
                              data = df2_trainingset,
                              method = 'lm',
                              trControl = trc)
mod2stpw_trainRMSEtrc 
# RMSE for mod2stpw_trainRMSE = 315

# 6. Create final model after cross validation
final_mod2_fwd <- lm(sales ~ region 
                     + Apr 
                     + prom 
                     + tv
                     + price
                     + banner
                     + May
                     + Jun
                     + Dec
                     + Nov
                     + Aug
                     + Mar,
                 data = df2_trainingset)

final_mod2_stpw <- lm(sales ~ region 
                      + Apr 
                      + prom 
                      + tv
                      + price
                      + banner
                      + May
                      + Jun
                      + Dec
                      + Nov
                      + Aug,
                     data = df2_trainingset)
summary(final_mod2_stpw)

# 7. CV test set with training model
mod2fwd_testpredict <- predict.lm(object = final_mod2_fwd,newdata = df2_testset)
mod2stpw_testpredict <- predict.lm(object = final_mod2_stpw,newdata = df2_testset)

# 8. Calculating RMSE for test set
mod2fwd_testMSE = sum( (df2_testset[,"sales"]-mod2fwd_testpredict)^2 ) / nrow(df2_testset)
mod2fwd_testMSE
mod2fwd_testRMSE <- sqrt(mod2fwd_testMSE)
mod2fwd_testRMSE

# RMSE for mod2fwd_testRMSE = 324

mod2stpw_testMSE = sum( (df2_testset[,"sales"]-mod2stpw_testpredict)^2 ) / nrow(df2_testset)
mod2stpw_testMSE
mod2stpw_testRMSE <- sqrt(mod2stpw_testMSE)
mod2stpw_testRMSE
# RMSE for mod2stpw_testRMSE = 323

### RESULT SHOWS THAT USING MODEL AFTER VARIABLE SELECTION, THE RMSE ARE:
### TRAIN (fwd) = 315
### TEST (fwd) = 324
### TRAIN (stpw) = 315
### TEST (stpw) = 323

########## Plotting for Model 2 ##########
# Diagnostic Plot
autoplot(final_mod2, 
         label.size = 3, 
         colour = 'steelblue',
         smooth.colour = 'red', 
         smooth.linetype = 'solid',
         ad.colour = 'green') + 
         theme_minimal() 

# Plotting linear relationships for sales and mkt variables (tv, banner, prom) 
ggplot(df2, aes(x=tv, y=sales)) +
  geom_point(color='skyblue4') +
  geom_smooth(method=lm, color='red') +
  ggtitle('Sales ~ TV') + 
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5))

ggplot(df2, aes(x=banner, y=sales)) +
  geom_point(color='seagreen') +
  geom_smooth(method=lm, color='darkorange') +
  ggtitle('Sales ~ Banner') + 
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5))

ggplot(df2, aes(x=prom, y=sales)) +
  geom_point(color='orchid') +
  geom_smooth(method=lm, color='dodgerblue') +
  ggtitle('Sales ~ Prom') + 
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5))

# Plot the affect of marketing activities on sales
ggplot(df2, aes(Marketing, sales, label = rownames(df2))) +
       geom_bar(stat='summary', fun.y='mean',fill='red') +
       geom_hline(yintercept = 4300, linetype = 'dashed', color ='blue') +
       ggtitle('No Marketing vs Marketing') + 
       theme_minimal() + 
       theme(plot.title = element_text(hjust = 0.5))
       

# Plot Residual Histogram (df2. Sales ~ Predicted Sales)
df2_residuals<- df2$sales - predict.lm(object = final_mod2,newdata = df2)

hist(df2_residuals,bty="n",
     main="Residual Histogram",
     pch=18,
     col="aquamarine",
     cex.axis=2,
     cex=1,
     cex.main=3,
     cex.lab=3,
     freq = FALSE,
     breaks=20)
     points(x = seq(min(df2_residuals),
            max(df2_residuals),
            length.out = 100), 
            dnorm(x = seq(min(df2_residuals),
                  max(df2_residuals), 
                  length.out = 100),
                  mean = 0, sd = sd(df2_residuals)), 
                  col="red",type="l")
     
# Statistical test to check residual normality
shapiro.test(df2_residuals) # p-value = 0.2 > 0.05. Residual normally distributed

### Plotting prediction ~ residual plot for train and test set
# Train model residual plot
dfrsd_trainingset <- df2_trainingset
dfrsd_trainingset$prediction <- predict.lm(object = final_mod2_stpw,newdata = df2_trainingset)
dfrsd_trainingset$residuals <- dfrsd_trainingset$sales - predict.lm(object = final_mod2_stpw,newdata = df2_trainingset)

train_prediction <- ggplot(dfrsd_trainingset, 
                           aes(x = prediction,
                           y = sales)) +
                           geom_point(color = 'darkslategray') + 
                           geom_abline(color = 'firebrick') + 
                           ggtitle('Training Set') + 
                           theme_minimal() +
                           theme(plot.title = element_text(hjust = 0.5))

#plot(train_prediction)

train_residuals <- ggplot(data = dfrsd_trainingset,
                          aes(x = prediction,
                          y = residuals)) + 
                          geom_pointrange(aes(ymin = 0,
                          ymax = residuals),
                          color = 'darkslategray') +
                          geom_hline(yintercept = 0,
                          linetype = 1,
                          color = 'firebrick') + 
                          ggtitle('Training Set') + 
                          theme_minimal() + 
                          theme(plot.title = element_text(hjust = 0.5))

#plot(train_residuals)

multiplot(train_prediction, train_residuals)

# Test model residual plot
dfrsd_testset <- df2_testset
dfrsd_testset$prediction <- predict.lm(object = final_mod2_stpw,newdata = df2_testset)
dfrsd_testset$residuals <- dfrsd_testset$sales - predict.lm(object = final_mod2_stpw,newdata = df2_testset)

test_prediction <- ggplot(dfrsd_testset, 
                          aes(x = prediction,
                          y = sales)) +
                          geom_point(color = 'steelblue') + 
                          geom_abline(color = 'firebrick') + 
                          ggtitle('Test Set') + 
                          theme_minimal() +
                          theme(plot.title = element_text(hjust = 0.5))

#plot(test_prediction)
                          
test_residuals <- ggplot(data = dfrsd_testset,
                          aes(x = prediction,
                          y = residuals)) + 
                          geom_pointrange(aes(ymin = 0,
                          ymax = residuals),
                          color = 'steelblue') +
                          geom_hline(yintercept = 0,
                          linetype = 1,
                          color = 'firebrick') + 
                          ggtitle('Test Set') + 
                          theme_minimal() +
                          theme(plot.title = element_text(hjust = 0.5))

#plot(test_residuals)

grid.arrange(train_prediction, train_residuals, test_prediction, test_residuals)

# Find Mean Absolute Percentage Error for Train & Test Set
mape(dfrsd_testset$sales,dfrsd_testset$prediction) #8.9%
mape(dfrsd_trainingset$sales,dfrsd_trainingset$prediction) #6.5%
