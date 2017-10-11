# lstm_time_series

EXECUTIVE SUMMARY

Use LTSM (RNN) to model airline traffic as a time series and forecast with 36 month horizon.


INTRODUCTION

Airlines, airport operators, taxi operators and hotels are interested in air passenger
forecasting.
Air passenger forecasting is a subset of a field known as time series analysis.
Historically,time series analysis has used techniques such as exponential moving
average and ARIMA.
After the advent of machine learning, recurrent neural networks (RNNs) have become
important as well.

In this project, we will discuss how an RNN can be used to solve an air passenger
forecasting problem.

PROBLEM STATEMENT

Our dataset is titled "International Airline Passengers: Monthly Totals January 1949---
December 1960" and is from the book "Time Series Analysis Forecasting and Control" by
Box & Jenkins.  Here is a partial listing of the dataset:

  Month       Air Passengers
  
  Jan 1949    112,000
  
  Feb 1949    118,000
  
  Mar 1949    132,000
  
  ....................
  
  Dec 1960    432,000
  
Let us imagine ourselves to be in January 1957; we want to forecast how many
air passengers there will be in January 1960, 36 months from now.
(Of course that number is in the 109th row of the dataset, but we will not peek at the
109th row---we want to compute the number by means of an RNN!)

After this let us imagine ourselves to be in February 1957; we want to forecast
how many air passengers there will be in February 1960, 36 months from now.

After that let us imagine ourselves to be in March 1957; we want to forecast
how many air passengers there will be in March 1960, 36 months from now.

A MORE TECHNICAL WAY OF STATING THE PROBLEM

The first 108 rows of the dataset are the training data; the remaining 36 rows are the
testing set.  We will use an RNN---more specifically the Keras package's LSTM model---
in order to compute the testing data; this computation will constitute our forecast.

HOW TO RUN THE CODE

Create a directory named, say, lstm_time_series.

Download the files airline.csv and cs_airline.py into that directory.

Type python cs_airline.py

OUTPUT

The output comprises several .png files.  Each .png file is a plot of the accuracy of
the forecast.  Why are there several forecasts?  Because the number of hidden neurons is
an adjustable parameter.  One of the forecasts was generated with 32 hidden neurons,
another with 64 hidden neurons, yet another with 128 hidden neurons, and so on.

WHAT I FOUND MOST INTERESTING

The dataset has a linear trend and a seasonal trend.  The code does not attempt to handle
the two trends separately.  Rather the code tries to learn everything.  To be effective, it
needs 128--256 hidden neurons.

Is 128 hidden neurons a lot?  Well, let's look at comparables.

Some authors use 150-300 hidden neurons to model a sine wave with RNNs.  So I suggest that
anything that has a periodic (or seasonal) component will need that many hidden neurons.

There is indeed an author who attacked the same air passenger forecasting problem using
only 4 hidden neurons!  How did he get so low a figure? It turned out that he handled the
linear and seasonal trends in analytical manner.  That is to say his RNN only had to learn
on what was left in the data after the linear and seasonal trends were analytically handled!

FOR MORE INFO

Please download the .pdf file in this repository
