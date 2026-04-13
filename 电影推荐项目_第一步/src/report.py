from pathlib import Path



def generate_explanation_sentences(global_stats: dict) -> list[str]:
    explanations = []

    sparsity = global_stats["matrix_sparsity"]
    if sparsity >= 0.9:
        explanations.append(
            f"- The user-item rating matrix sparsity is {sparsity:.4f}, which indicates that the data is highly sparse. This may weaken similarity estimation in collaborative filtering."
        )
    else:
        explanations.append(
            f"- The user-item rating matrix sparsity is {sparsity:.4f}. Sparsity still exists, but it is relatively manageable."
        )

    long_tail_ratio = global_stats["top_20_percent_rating_share"]
    if long_tail_ratio >= 0.6:
        explanations.append(
            f"- The top 20% of movies contribute {long_tail_ratio:.2%} of all ratings, showing a clear long-tail effect and strong popularity concentration."
        )
    else:
        explanations.append(
            f"- The top 20% of movies contribute {long_tail_ratio:.2%} of all ratings, indicating a moderate concentration of popularity."
        )

    rating_mean = global_stats["rating_mean"]
    if rating_mean >= 3.5:
        explanations.append(
            f"- The average rating is {rating_mean:.2f}, suggesting that users tend to rate movies positively overall."
        )
    else:
        explanations.append(
            f"- The average rating is {rating_mean:.2f}, suggesting that user ratings are relatively conservative."
        )

    user_median = global_stats["user_rating_count_median"]
    if user_median < global_stats["user_average_rating_count"]:
        explanations.append(
            "- The median number of ratings per user is lower than the mean, which means a small number of highly active users raise the overall average. User activity is unevenly distributed."
        )
    else:
        explanations.append("- The user rating count distribution is relatively balanced.")

    item_median = global_stats["item_rating_count_median"]
    if item_median < global_stats["item_average_rating_count"]:
        explanations.append(
            "- The median number of ratings per movie is lower than the mean, which means a small number of popular movies receive a large share of exposure."
        )
    else:
        explanations.append("- The movie popularity distribution is relatively balanced.")

    explanations.append(
        "- The next step should be to build baseline models first, then compare collaborative filtering and matrix factorization methods, with additional subgroup analysis for active users and cold items."
    )
    return explanations



def generate_analysis_report(output_dir: Path, global_stats: dict):
    file_path = output_dir / "step1_analysis_summary.md"
    explanation_sentences = generate_explanation_sentences(global_stats)

    content = f"""# Step 1 Analysis Summary

## 1. Core Statistics

- Number of users: {global_stats['num_users']}
- Number of movies: {global_stats['num_items']}
- Number of ratings: {global_stats['num_ratings']}
- Mean rating: {global_stats['rating_mean']}
- Median rating: {global_stats['rating_median']}
- Rating standard deviation: {global_stats['rating_std']}
- Matrix sparsity: {global_stats['matrix_sparsity']}
- Average number of ratings per user: {global_stats['user_average_rating_count']}
- Median number of ratings per user: {global_stats['user_rating_count_median']}
- Average number of ratings per movie: {global_stats['item_average_rating_count']}
- Median number of ratings per movie: {global_stats['item_rating_count_median']}
- High activity user ratio: {global_stats['high_activity_user_ratio']:.2%}
- Top 20% movie count: {global_stats['top_20_percent_item_count']}
- Rating share from top 20% movies: {global_stats['top_20_percent_rating_share']:.2%}

## 2. Observations

{chr(10).join(explanation_sentences)}

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
"""

    file_path.write_text(content, encoding="utf-8")
