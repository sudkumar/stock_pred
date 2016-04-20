library(forecast)
library(caret)
library(quantmod)

getSymbols("MSFT",src="yahoo",from = "2007-01-01",
           to = Sys.Date())
data=MSFT
data=as.data.frame(data)
date=rownames(data)

close=data$MSFT.Close
stclose=(close-mean(close))/sd(close)
################################################
################################################
log_returns <- diff(log(MSFT$MSFT.Close), lag=1)
log_returns=na.omit(log_returns)
y=auto.arima(log_returns[2:2000])
plot(forecast(y,h=60),xlim=c(2200,2350),main="Forecast by ARIMA(2,0,0)")

f=forecast(y,h=300)
p=f$mean
RMSE=sqrt(sum(as.data.frame(log_returns[2001:2300])[,1]-p)^2)/300

fit <- nnetar(log_returns)
plot(forecast(fit,h=60),xlim=c(2200,2350),main="Forecast by Neural Networks")
points(1:length(x),fitted(fit),type="l",col="green")
f1=forecast(fit,h=300)
p1=f1$mean
RMSE=sqrt(sum(as.data.frame(log_returns[2001:2300])[,1]-p1)^2)/300

library(ggplot2)
dat <- data.frame(c=timex[2001:2300],a=p,b=p1)
ggplot(dat) + geom_path(aes(x=c,y=a),col=I("red")) + geom_path(aes(x=c,y=b),col=I("blue"))





T=data.frame(a=date)
T$Date=as.Date(T$a, "%Y-%m-%d")
Diff <- function(x, start) as.numeric(x - as.Date(cut(start, "year")))
time=transform(T, NumDays = Diff(Date, Date), TotalDays = Diff(Date, Date[1]))
timex=time[,4]
log_returns1=as.data.frame(log_returns)[2:2337,1]
library(kernlab)
foo <- gausspr(timex[1:2000], as.data.frame(log_returns1[1:2000]))
foo
# predict and plot
ytest <- predict(foo, timex[2000:2300])

predicted=c(log_returns1[1:1999],ytest)



library(ggplot2)
dat <- data.frame(c=timex[1:2300],a=log_returns1[1:2300],b=predicted)
ggplot(dat) + geom_path(aes(x=c,y=a),col=I("red")) + geom_path(aes(x=c,y=b),col=I("blue"))

dat <- data.frame(c=timex[2000:2300],a=log_returns1[2000:2300],b=predicted[2000:2300])
ggplot(dat) + geom_path(aes(x=c,y=a),col=I("red")) + geom_path(aes(x=c,y=b),col=I("blue"))




GPmodel = GP_fit(timex[1:2000], stclose[1:20])
GPprediction = predict.GP(GPmodel,timex[21:30]);
yhat = GPprediction$Y_hat;
mse = GPprediction$MSE;
completedata = GPprediction$complete_data;
completedata;
