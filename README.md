

```python
# Airbnb data analysis project

## Table of contents

* [Project Goals](#Project-goals)
* [Data Sources](#Data-sources)
* [Built with](#Built-with)
* [Project Steps](#Project-steps)
* [Author](#Author)

## Project Goals

The project goal was to write a 'data science blog post' about Airbnb listings in Boston and Seattle that answers three questions.
The data preparation and analysis was conducted to provide answers to the below:
* To which extent is data available on listings of Boston and Seattle Aribnbs? Is there a significant difference in the availability of data for the two cities?
* Does more information provision correlate with more favorable outcomes for a host? E.g. better prices, more reviews or better review score values
* Are there significant differences between the two cities’ apartment offers?

## Data Sources

-	Seattle Airbnb Open Data: https://www.kaggle.com/airbnb/seattle, retreived on September 4, 2020
-	Boston Airbnb Open Data: https://www.kaggle.com/airbnb/boston, retreived on September 4, 2020

## Built with

The project was primarily built in Python.
- The script was build in a Jupyter Notebook loaded from Anaconda Navigator.
- Key Python packages used: pandas, numpy, matplotlib, seaborn. 
- An additional dashboard was prepared in Tableau Online.
- The blog post was written on Medium.

## Project Steps

1. __Business understanding__
* Based on the provided datasets, I compiled three questions to be answered by data analysis.

2. __Data understanding__
* Data was loaded.
* General information was assessed such as the shape of the datasets and datatypes.

3. __Data preparation__
* Datasets became joint and concatenated.
* Price data got cleaned.
* I dropped feature for which more than 60% of the records were missing.
* I dropped rows which contained more than 40 missing values (out of cca. 95 originally).
* I changed the data type of ID features in order to avoid their inclusion in later numeric analysis.

4. __Data analysis and visualization__
* Scatter plots were utilized to identify patterns between missing data volume and target variables such as price or number of reviews.
* Correlations between numeric variables were identified by a correlation matrix heatmap.
* Bar charts and histograms were prepared to compare Boston and Seattle data.

5. __Evaluation__
* Findings from each parts of the analysis were summarized.

## Author

Eszter Rékasi - [LinkedIn profile](https://www.linkedin.com/in/eszter-r%C3%A9kasi/)

```
