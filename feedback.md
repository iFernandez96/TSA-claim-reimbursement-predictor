---

**Problem 1.1: Title/Authors/Sections (Grade: A-)**

**Comments:**
- Overall, you are making good progress.
- The paper is not itself a predictor, so I recommend changing the title. One option could be something like:  
  *"Predicting the Amount of a TSA Claim Reimbursement"*

---

**Problem 1.2: Introduction (Grade: B)**

**Comments:**
- The introduction should consist of two paragraphs:
  1. One paragraph discussing the purpose of the report.
  2. Another paragraph describing the dataset.

- Some of this information currently appears at the top of the Data Exploration section. The link to the dataset should be included in the introduction when the dataset is first mentioned.

- Remove "Objective:" from the beginning of the intro. Instead, start with something like:  
  *"In this notebook, we report on our work to predict..."*

- Do not assume the reader knows what TSA stands for; expand the acronym in the introduction.

- Provide more high-level information about the dataset. For example, clarify if the claims involve only damaged items, and whether these items were damaged on the plane, in the airport, or elsewhere.

- Note that the goal of the report is not to write an algorithm, but to document and explain the predictive process.

---

**Problem 1.3: Exploration and Visualization (Grade: A)**

**Comments:**
- The first plot is currently showing fractional amounts, not sums. Clarify this in your description.
- It is misleading to say that the first plot represents the entire dataset. Instead, say that the plot shows the distribution of items for which claims are made.

---

**Problem 1.4: Preprocessing (Grade: B)**

**Comments:**
- Commentary about preprocessing steps should be in Markdown cells, not in code cell comments.
- Remove extra blank lines from code cells.
- Consider whether removing airline names before visualization is a good idea.
- Preprocessing should include an assessment of missing data.
- In the initial exploration, focus on key variables such as Claim Amount and Close Amount.
- The mean claim and close amount plot should be explained better.  
  - Address why there are no visible orange bars.  
  - Verify the data; is a $140,000 average claim for medicine realistic?

- If you choose to use +/- three standard deviations for data cleaning, first examine the distribution of claim and close amounts. A safer approach might be to remove rows in the top/bottom 1% or 0.5% of values.

- Explain what "claim amount" and "close amount" mean.
- Avoid informal commentary like "Now we are getting somewhere" or "Great!"—this is not appropriate for a professional report.
- When visualizing percentages of claims, consider using a multi-box plot rather than just showing the average.

---

**Problem 1.5: Machine Learning (ID: 38213, Fernandezisrael) (Grade: A-)**

**Comments:**
- Compute the baseline from the training data.
- Do not use the test data until the end of the report. Use cross-validation for model development.
- Include additional predictor variables to improve the model.

---

**Problem 1.Participation: 0**

---
