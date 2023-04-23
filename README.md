# Amazon Products Reviews Classification

In this task, we will perform text classification on a set of reviews from three diverse domains. The goal is to predict the review's score (rating) based on its text, which ranges from 1-5, resulting in a 5-way classification. We will be using the train and test datasets, each containing 2000 and 400 reviews per class, respectively. We will train a classifier on the train data and evaluate and report the results on the test data.

## Input
Each review is presented in the following format:
```
{
"overall": 4.0,
"verified": true,
"reviewTime": "09 7, 2015",
"reviewerID": "A2HDTDZZK1V4KD",
"asin": "B0018CLO1I",
"reviewerName": "Rachel E. Battaglia",
"reviewText": "This is a great litter box! The door rarely gets stuck, it's easy to clean, and my cat likes it. I took one star off because the large is really small. My cat is 6 pounds and this isn't quite big enough for her. I'm ordering this same brand in a large. Online price is much cheaper than pets stores or other stores!",
"summary": "Great Box, Get XL size!",
"unixReviewTime": 1441584000
}
```

where:
- `overall` refers to the rating, which is the label we are trying to predict
- `reviewText` refers to the body of the review
- `summary` refers to the summary of the review

## Output
The output will be a dictionary that contains the F1 metric per each class, as well as the overall accuracy. The goal is to achieve the highest possible accuracy on the classification task.
