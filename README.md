# Customer Segmentation and Acquisition - Bertelsmann Arvato
## Machine Learning Engineer Udacity Nanodegree - Capstone Project

This repository contains code and report for "Capstone Project - Arvato Customer Segmentation" done as part of Udacity Machine Learning Engineer Nanodegree program.


## Table of Contents

- [Project Overview](#projectoverview)
- [Data Description](#datadescription)
- [Technical Overview](#technicaloverview)
- [Requirements](#requirements)
- [Results](#results)

***

<a id='projectoverview'></a>
## Project Overview

In this project, the demographic data of German population and the customer data have been analysed in order to perform Customer Segmentation and Customer Acquisition. Arvato Financial Solutions is a services company that provides financial services, Information Technology (IT) services and Supply Chain Management (SCM) solutions for business customers on a global scale.

This project is to help a Mail-Order company to acquire new customers to sell its organic products. The goal of this project is to understand the customer demographics as compared to general population in order to decide whether to approach a person for future products.

This project is divided into two steps:

1. `Customer Segmentation using Unsupervised Learning`, in this part a thorough data analysis and feature engineering steps are performed to prepare the data for further steps. A Principal Component Analysis (PCA) is performed for dimensionality reduction. Then K-Means Clustering is performed on the PCA components to cluster the general population and the customer population into different segments. These clusters are studied to determine what features make a customer with the help of cluster weights and component weights.

2. `Customer Acquisition using Supervised Learning`, in this part of the project the customers data with defined targets indicating the past responses of the customers has been used to train Supervised Learning algorithms. Then the trained model is used to make predictions on unseen test data to determine whether a person could be a possible customer.

<a id='datadescription'></a>
## Data Description

The data has been provided by Udacity and Arvato Financial Solutions. The dataset contains 4 data files and 2 description files. The description files have information about the features and their explanation.
The 4 data files include:
* Customer Segmentation
  * General Population demographics
  * Customer demographics
* Customer Acquisition
  * Training data
  * Test data

<a id='technicaloverview'></a>
## Technical Overview

The project has been divided into various steps which include:
* Data Exploration and Cleaning
* Feature Engineering
* Dimensionality Reduction
* Clustering
* Supervised Learning
* Model Evaluation
* Predictions on Test data
* Submission to Kaggle and Scoring

An explanation about each step and choice of algorithms, metrics has been given in the `Report.pdf`.


<a id='requirements'></a>
## Requirements

All of the requirements are given in requirements.txt. To install Run: `pip install -r requirements.txt`


<a id='results'></a>
## Results

The results have been clearly documented in the Jupyter Notebook. Please refer [Arvato Project Workbook.ipynb](https://github.com/pranaymodukuru/Bertelsmann-Arvato-customer-segmentation/blob/master/Arvato%20Project%20Workbook.ipynb).
