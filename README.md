# Student Performance Analysis and Prediction

> **Exploratory Data Analysis
> (EDA) and Machine Learning**

A beginner-friendly data science project that explores student
performance in Mathematics and Portuguese, visualizes patterns in the
data, and uses **Linear Regression** to predict students' final grades.

------------------------------------------------------------------------

## Table of Contents

-   [Project Overview](#project-overview)
-   [Objectives](#objectives)
-   [Why Linear Regression?](#why-linear-regression)
-   [Datasets](#datasets)
-   [Technology Stack](#technology-stack)
-   [Project Workflow](#project-workflow)
-   [Exploratory Data Analysis and
    Screenshots](#exploratory-data-analysis-and-screenshots)
-   [Machine Learning Approach](#machine-learning-approach)
-   [Model Evaluation Results](#model-evaluation-results)
-   [How to Run the Project](#how-to-run-the-project)
-   [Key Findings](#key-findings)
-   [Limitations and Future
    Improvements](#limitations-and-future-improvements)
-   [Project Structure](#project-structure)
-   [Acknowledgements](#acknowledgements)

------------------------------------------------------------------------

## Project Overview

Student performance can be influenced by academic history, study habits,
family background, and other factors. This project explores student
records to understand grade distributions and relationships between
selected variables, then trains a regression model to estimate the final
grade (`G3`).

Two files are used:

-   `Maths.csv` --- student information and Mathematics grades.
-   `Portuguese.csv` --- student information and Portuguese language
    grades.

The two datasets are combined for a subject-level comparison. A `course`
column identifies whether each record belongs to Maths or Portuguese.

**Prediction target:** `G3`, the final grade on a 0--20 scale.

> **Dataset distinction:** The Week 3 brief describes a different
> Student Performance dataset with `Hours_studied`, `Previous_scores`,
> `Sleep_hours`, `sample_question_papers_practiced`, and
> `performance_index`. This project adapts the Week 3 regression
> workflow to the uploaded Maths and Portuguese files. Its target is
> `G3`, not `performance_index`.

## Objectives

1.  Load and inspect the Maths and Portuguese datasets using Python.
2.  Check data types, missing values, and duplicate records.
3.  Combine the datasets while retaining the course label.
4.  Perform Exploratory Data Analysis (EDA).
5.  Visualize final-grade distributions and relationships with selected
    variables.
6.  Prepare numerical and categorical features for machine learning.
7.  Split the data into training and testing sets.
8.  train a Linear Regression model with Scikit-learn.
9.  Predict final grades and evaluate the predictions using MAE, MSE,
    RMSE, and R².
10. Document observations, results, limitations, and potential next
    steps.

## Why Linear Regression?

Linear Regression is a useful first model for this project because:

-   **The target is numeric:** `G3` is a numerical grade, so this is a
    regression problem rather than a classification problem.
-   **It is a clear baseline:** The model provides a straightforward
    starting point before trying more complex algorithms.
-   **It is relatively easy to explain:** It estimates a relationship
    between input features and a continuous target.
-   **It works with Scikit-learn pipelines:** Categorical columns can be
    encoded and passed to the model in a repeatable workflow.
-   **It can be evaluated directly:** Metrics such as MAE, RMSE, and R²
    help describe the size of errors and how much variation the model
    explains.

Linear Regression was chosen for learning and as a baseline; it is **not
guaranteed to be the best model** for this dataset. Its assumptions and
performance should be compared with alternatives in future work.

## Datasets

### Files

  File               Purpose
  ------------------ -------------------------------------------------
  `Maths.csv`        Student attributes and Maths course grades
  `Portuguese.csv`   Student attributes and Portuguese course grades

The files include student attributes such as age, study time, past
failures, family education, absences, and grades.

### Important columns

  Column        Description
  ------------- ----------------------------------------------
  `studytime`   Weekly study-time category
  `failures`    Recorded past class-failure category
  `G1`          First-period grade
  `G2`          Second-period grade
  `G3`          Final grade; the prediction target
  `course`      Added project label: `Maths` or `Portuguese`

`G1` and `G2` are earlier-period grades. The model in this project uses
the available columns other than `G3` as features, so it includes `G1`
and `G2`. This is appropriate only when those earlier grades are already
available at prediction time.

## Technology Stack

-   **Python** --- programming language
-   **Pandas** --- data loading, cleaning, and manipulation
-   **NumPy** --- numerical operations
-   **Matplotlib** --- plotting
-   **Seaborn** --- statistical visualizations
-   **Scikit-learn** --- data splitting, preprocessing, Linear
    Regression, and evaluation
-   **Google Colab / Jupyter Notebook** --- development environment

## Project Workflow

``` text
Maths.csv + Portuguese.csv
            |
            v
      Load the data
            |
            v
 Inspect columns, types, and shape
            |
            v
 Check missing values and duplicates
            |
            v
 Add course label and combine datasets
            |
            v
 Exploratory Data Analysis (EDA)
            |
            v
 Select X features and y = G3
            |
            v
 Train-test split (80% / 20%)
            |
            v
 Encode categorical features
            |
            v
 Train Linear Regression model
            |
            v
 Predict grades on the test set
            |
            v
 Evaluate MAE, MSE, RMSE, and R²
            |
            v
 Interpret findings and limitations
```

### Data preparation

-   Duplicate rows were removed from each dataset.
-   Rows with a missing `G3` target were removed.
-   The datasets were combined and labelled by course.
-   Numerical and categorical predictors were identified separately.
-   Categorical predictors were transformed using `OneHotEncoder` with
    `handle_unknown="ignore"`.
-   An 80/20 train-test split was used with `random_state=42`.
-   Preprocessing and regression were placed in a Scikit-learn
    `Pipeline`.

The train-test split helps evaluate the model on rows that were not used
to fit it. The `random_state` makes the split reproducible.

## Exploratory Data Analysis and Screenshots

The screenshots below were captured from the project workflow. Values
described as "approximately" are visual estimates from the charts; use
calculated summaries for exact statistics.

### 1. Distribution of Final Student Grades

<img width="1042" height="615" alt="image" src="https://github.com/user-attachments/assets/6897e218-5dee-4a3a-aa26-87d3df4430df" />


**What the chart shows**

-   The x-axis is the final grade (`G3`), from 0 to 20.
-   The y-axis is the number of students in each grade interval.
-   Blue represents Maths; orange represents Portuguese.
-   The smooth lines show the estimated distribution shape.

**Observation**

Many grades in both subjects are concentrated around the middle of the
scale, particularly roughly 10--15. The chart also shows a noticeable
group of Maths records at grade 0. Investigate these records rather than
automatically deleting them. Bar heights should not be used alone to
conclude which subject performed better because the datasets may contain
different numbers of students.

### 2. Final Grade Comparison by Course

<img width="846" height="596" alt="image" src="https://github.com/user-attachments/assets/0d02ecd3-1428-4e81-a55c-6576658dc7c0" />


**What the chart shows**

A box plot compares the median, middle 50% of grades, spread, and
potential outliers for each course.

**Observation**

The plotted median is approximately 11 for Maths and 12 for Portuguese.
The middle 50% of Maths grades appears more spread out, while Portuguese
includes very low potential outliers around 0--1. These are descriptive
observations; this plot alone does not establish that the difference
between subjects is statistically significant.

### 3. Study Time vs Final Grade

<img width="897" height="597" alt="image" src="https://github.com/user-attachments/assets/a99528a9-ae3a-4786-b808-df9b30dd7355" />


**What the chart shows**

The x-axis uses the four weekly study-time categories:

-   `1`: less than 2 hours
-   `2`: 2--5 hours
-   `3`: 5--10 hours
-   `4`: more than 10 hours

The y-axis shows final grade (`G3`).

**Observation**

The median grade appears highest in category 3 (approximately 13).
Category 4 has a median around 12, so the plot does not show a steadily
increasing median for every increase in study-time category. There is
variation within each category, which means study time alone cannot
explain students' grades. This is association, not proof of causation.

### 4. Past Class Failures vs Final Grade

<img width="902" height="591" alt="image" src="https://github.com/user-attachments/assets/3836d798-b7a3-4cba-98ff-0d129cfd4b7d" />


**What the chart shows**

The plot compares final grades across the recorded `failures`
categories.

**Observation**

The median appears highest for category 0 (approximately 12), lower for
category 1 (approximately 10), and around 8 for categories 2 and 3. This
pattern suggests that recorded past failures are associated with lower
final grades in this dataset.

**Data interpretation caution:** The original dataset's `failures` field
has a specific coding convention. Verify the raw values and any
transformations before describing the plotted categories as exact
numbers of failures. The plot alone does not prove that failures cause
lower grades.

### 5. Correlation Heatmap Between Numerical Variables

<img width="935" height="651" alt="image" src="https://github.com/user-attachments/assets/5c9e1c42-3a5c-4a3e-a1e2-1b36027109e1" />


**What the chart shows**

Each cell displays the correlation between two numerical variables.
Values range from -1 to +1:

-   Positive values: variables tend to increase together.
-   Negative values: one tends to decrease as the other increases.
-   Values near zero: little linear relationship is visible.

**Important observed correlations**

  ------------------------------------------------------------------------
  Variable pair                    Correlation shown Interpretation
  --------------------- ---------------------------- ---------------------
  `G2` and `G3`                                 0.91 Strong positive
                                                     relationship

  `G1` and `G2`                                 0.86 Strong positive
                                                     relationship

  `G1` and `G3`                                 0.81 Strong positive
                                                     relationship

  `Medu` and `Fedu`                             0.64 Positive relationship
                                                     between parents'
                                                     education levels

  `Dalc` and `Walc`                             0.63 Positive relationship
                                                     between workday and
                                                     weekend
                                                     alcohol-consumption
                                                     ratings

  `failures` and `G3`                          -0.38 Negative relationship

  `studytime` and `G3`                          0.16 Weak positive
                                                     relationship

  `absences` and `G3`                          -0.05 Very weak linear
                                                     relationship
  ------------------------------------------------------------------------

**Observation**

Earlier-period grades, especially `G2`, are strongly associated with the
final grade. Past failures have a negative correlation with `G3`, while
study time has a weak positive correlation. Correlation does not prove
causation.

### 6. Actual vs Predicted Grades --- Sample Rows

<img width="446" height="460" alt="image" src="https://github.com/user-attachments/assets/51e61bc8-d6ea-45e4-96eb-121a0474b728" />



**What the output shows**

The table compares actual grades in the test set with predictions from
the trained model. For example, one row has an actual grade of 11 and a
predicted grade of 10.47; another has an actual grade of 15 and a
predicted grade of 14.65.

**Observation**

Several sample predictions are close to their actual grades. The row
with an actual grade of 0 and a predicted grade of 5.57 shows a larger
error. These ten rows are examples only; the evaluation metrics must be
calculated over the full test set.

### 7. Model Evaluation Results

<img width="542" height="177" alt="image" src="https://github.com/user-attachments/assets/7c1069c7-ee21-4507-abd2-8c142a318393" />


The recorded test-set metrics are:

  ------------------------------------------------------------------------
  Metric                                      Result Meaning
  --------------------- ---------------------------- ---------------------
  Mean Absolute Error                       **1.02** Average absolute
  (MAE)                                              prediction error, in
                                                     grade points

  Mean Squared Error                        **3.10** Average squared error
  (MSE)                                              

  Root Mean Squared                         **1.76** Error measure in
  Error (RMSE)                                       grade points that
                                                     penalizes larger
                                                     errors more

  R²                                      **0.7997** Proportion of
                                                     test-set variation
                                                     explained relative to
                                                     a mean-prediction
                                                     baseline

  R² as a percentage                      **79.97%** Another way to
                                                     express the R² value;
                                                     not classification
                                                     accuracy
  ------------------------------------------------------------------------

**Observation**

The MAE of 1.02 means predictions differ from actual grades by about
1.02 grade points on average in absolute terms. The RMSE of 1.76 is
higher, reflecting the greater impact of larger errors. R² of 0.7997
indicates that the model explains about 79.97% of the variation in
test-set grades under this split.

**Important:** R² is not the percentage of predictions that are correct,
and regression should not be described using classification accuracy.

### 8. Actual vs Predicted Student Grades

<img width="872" height="651" alt="image" src="https://github.com/user-attachments/assets/4006fd0b-adb3-44da-88e0-ad4d7e57ea3e" />

**What the chart shows**

-   The x-axis contains actual grades.
-   The y-axis contains predicted grades.
-   The dashed diagonal line represents perfect predictions.
-   Points close to the line are more accurate; points above it are
    overpredictions, and points below it are underpredictions.

**Observation**

The points generally follow an upward trend and many lie near the
diagonal line, which is consistent with the positive R² result. Some
predictions differ more noticeably from the actual grades, particularly
for some records with an actual grade of zero. The model captures the
broad pattern but does not predict every student perfectly.

## Machine Learning Approach

### Target and features

-   **Target (`y`):** `G3`
-   **Features (`X`):** All remaining columns, including `G1`, `G2`,
    student attributes, and the added `course` label.

### Train-test split

The data was split into:

-   **80% training data** --- used to fit the model.
-   **20% test data** --- held back to evaluate predictions.

`random_state=42` was used so the split can be reproduced.

### Preprocessing and model

A Scikit-learn `ColumnTransformer` passes numerical features through and
applies `OneHotEncoder` to categorical features. A `Pipeline` combines
this preprocessing with `LinearRegression`.

### Prediction

The trained pipeline predicts grades for the test data. The actual and
predicted values are compared in a table and scatter plot.

## Model Evaluation Results

The model's recorded results are:

-   **MAE:** 1.02
-   **MSE:** 3.10
-   **RMSE:** 1.76
-   **R²:** 0.7997

Overall, the results suggest the model provides useful predictions on
this test split. These numbers should be reproduced by running the
notebook with the same data preparation, features, and split settings.
Results may differ if the data or preprocessing changes.

## How to Run the Project

### Option A --- Google Colab

1.  Open [Google Colab](https://colab.research.google.com/).

2.  Create a new notebook.

3.  Upload `Maths.csv` and `Portuguese.csv`.

4.  Install the required libraries if they are not already available:

    ``` python
    !pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
    ```

5.  Run the notebook cells in order: imports, loading, cleaning,
    combining, EDA, preprocessing, training, prediction, and evaluation.

6.  Save or download the notebook as an `.ipynb` file.

**File-format note:** The uploaded files were loaded in the working
notebook using `pd.read_excel(..., engine="openpyxl")` despite their
`.csv` filenames. If your local copies are genuine comma-separated CSV
files, use `pd.read_csv("Maths.csv")` and
`pd.read_csv("Portuguese.csv")` instead. If Excel loading raises an
error, inspect the actual file format and use the matching reader.

### Option B --- Local Python environment

1.  Install Python 3.

2.  Put the notebook and both datasets in the same project directory.

3.  Install dependencies:

    ``` bash
    pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter
    ```

4.  Start Jupyter:

    ``` bash
    jupyter notebook
    ```

5.  Open the notebook and run the cells from top to bottom.

## Key Findings

1.  Most final grades in the plotted distributions are concentrated
    around the middle of the 0--20 scale.
2.  Portuguese has a slightly higher displayed median than Maths, but a
    statistical test would be needed to assess whether a difference is
    significant.
3.  The study-time plot shows the highest median in category 3; the
    median does not increase monotonically across all four categories.
4.  Higher recorded past-failure categories are associated with lower
    median final grades in the displayed plot.
5.  `G2` and `G3` have a strong positive correlation (0.91), and `G1`
    and `G3` also have a strong positive correlation (0.81).
6.  The Linear Regression model achieved MAE 1.02, RMSE 1.76, and R²
    0.7997 on the reported test split.

## Limitations and Future Improvements

-   **Earlier grades are included:** `G1` and `G2` are strong predictors
    of `G3`. This is suitable only when those grades are available at
    the intended prediction time. For early-year prediction, train a
    second model without `G1` and `G2`.
-   **Compare models:** Evaluate alternatives such as Random Forest
    Regressor and Decision Tree Regressor, using the same train-test
    split.
-   **Use cross-validation:** A single split can produce a result that
    depends on which rows were held out. Cross-validation can provide a
    more stable estimate.
-   **Inspect grade-zero records:** Check their context and meaning
    before deciding whether any data-quality treatment is appropriate.
-   **Verify feature encoding:** Confirm the coding of `failures` and
    other ordinal columns before interpreting their numeric values as
    counts or equal intervals.
-   **Avoid causal claims:** Visualizations and correlations show
    associations, not proof that one factor causes a grade outcome.
-   **Consider the course label:** The model combines two subjects. A
    future experiment could compare subject-specific models or evaluate
    whether a shared model performs similarly for each course.
-   **Report reproducibility details:** Record the dataset versions,
    cleaning rules, random seed, feature list, and metrics with each
    experiment.

## Project Structure

``` text
Week-3-Student-Performance-Prediction/
├── README.md
├── Maths.csv
├── Portuguese.csv
└── screenshots/
    ├── 01-grade-distribution.png
    ├── 02-grade-comparison-by-course.png
    ├── 03-study-time-vs-final-grade.png
    ├── 04-past-failures-vs-final-grade.png
    ├── 05-numerical-correlation-heatmap.png
    ├── 06-actual-vs-predicted-table.png
    ├── 07-model-evaluation-metrics.png
    └── 08-actual-vs-predicted-plot.png
```
------------------------------------------------------------------------

**Author:** Suman Mandal

**Project type:** beginner machine learning project

**Primary task:** Regression --- predicting final student grades (`G3`)
