#Anthony Windmon - RF Classifier 
#References - YouTube, Stack Overflow, CRAN.r

library(randomForest)

#Loads dataset 
dataset = "C:\\Users\\awindmon\\Desktop\\DATASETS\\TEST_BEFORE_AFTER_COUGH_4416_CUT_TEST3_MFCC_ONLY.csv"

data <- read.csv(dataset, header=TRUE)

#shows the first few elements of the dataset
head(data)

#shows structure of dataset. Helps with pre-processing data.
str(data)

#show statistics of dataset 
summary(data)
summary(data$Class)
summary(data$MFCC1)

#scatterplot matric of dataset
plot(data)

#plots data by catergory (class)
#plot(data$Class)

#plots quantitative variables for MFCC1
plot(data$MFCC1)

#plots class and quan. Uses a box plot
plot(data$Class, data$MFCC1)

#plots two quantitative features
plot(data$MFCC1, data$MFCC11)

#Plot with different settings 
plot(data$MFCC2, data$MFCC3,
col = "#cc0000", #color for data (red)
pch = 19, #uses solid circles for colors
xlab = "MFCC-2",
ylab = "MFCC-3")

#plot bar graphs 
feature <- table(data$MFCC6)
barplot(feature)
plot(feature)

#Histogram
hist(data$MFCC1)
hist(data$MFCC2)
hist(data$MFCC3)
hist(data$MFCC4)

hist(data$MFCC1,
	breaks = 14, #14 bins 
	freq = FALSE, #Axis shows density, not freq
	col = "thistle1", #Color 
	main = paste("MFCC Data"), 
	xlab = "MFCC Raw Data") 

set.seed(42)
#Creates RF model and sets 'class' as the 'target' (python term)
model <- randomForest(Class ~., data=data, ntree=1000, proximity=TRUE)
model #displays the results


