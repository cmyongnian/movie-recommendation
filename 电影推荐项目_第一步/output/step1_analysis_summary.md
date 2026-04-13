# Step 1 Analysis Summary

## 1. Core Statistics

- Number of users: 943
- Number of movies: 1682
- Number of ratings: 100000
- Mean rating: 3.5299
- Median rating: 4.0
- Rating standard deviation: 1.1257
- Matrix sparsity: 0.936953
- Average number of ratings per user: 106.0445
- Median number of ratings per user: 65.0
- Average number of ratings per movie: 59.453
- Median number of ratings per movie: 27.0
- High activity user ratio: 33.93%
- Top 20% movie count: 337
- Rating share from top 20% movies: 64.72%

## 2. Observations

- The user-item rating matrix sparsity is 0.9370, which indicates that the data is highly sparse. This may weaken similarity estimation in collaborative filtering.
- The top 20% of movies contribute 64.72% of all ratings, showing a clear long-tail effect and strong popularity concentration.
- The average rating is 3.53, suggesting that users tend to rate movies positively overall.
- The median number of ratings per user is lower than the mean, which means a small number of highly active users raise the overall average. User activity is unevenly distributed.
- The median number of ratings per movie is lower than the mean, which means a small number of popular movies receive a large share of exposure.
- The next step should be to build baseline models first, then compare collaborative filtering and matrix factorization methods, with additional subgroup analysis for active users and cold items.

## 3. Implications for Modeling

1. Data sparsity can weaken similarity estimation in user-based and item-based collaborative filtering.
2. When movie popularity is heavily concentrated, models may become biased toward popular items, making cold-item recommendation more difficult.
3. Behavioral differences between highly active and less active users are significant, so subgroup evaluation should be considered in later experiments.
4. If matrix factorization is used later, it is often more stable than neighborhood methods in sparse settings.

## 4. Suggested Next Steps

1. Build baseline predictors using global average, user average, and item average.
2. Implement one collaborative filtering method.
3. Implement one matrix factorization method.
4. Compare results for overall users, low-activity users, and high-activity users.
