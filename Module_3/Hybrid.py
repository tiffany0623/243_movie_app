class HybridRecommender:
    MODEL_NAME = 'Hybrid'

    def __init__(self, cb_rec_model, cf_rec_model, items_df, cb_ensemble_weight=1.0, cf_ensemble_weight=1.0):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.cb_ensemble_weight = cb_ensemble_weight
        self.cf_ensemble_weight = cf_ensemble_weight
        self.items_df = items_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id,user_profile, items_to_ignore=[], topn=10, verbose=False):
        # Getting the top-1000 Content-based filtering recommendations
        cb_recs_df = self.cb_rec_model.recommend_items(user_id, user_profile=user_profile, items_to_ignore=items_to_ignore, verbose=verbose,
                                                       topn=1000).rename(columns={'recRating': 'recRatingCB'})

        # Getting the top-1000 Collaborative filtering recommendations
        cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                       topn=1000).rename(columns={'recRating': 'recRatingCF'})

        # Combining the results by movieId
        recs_df = cb_recs_df.merge(cf_recs_df,
                                   how='outer',
                                   left_on='movieId',
                                   right_on='movieId').fillna(0.0)

        # Computing a hybrid recommendation score based on CF and CB scores
        recs_df['recRatingHybrid'] = (recs_df['recRatingCB'] * self.cb_ensemble_weight) \
                                     + (recs_df['recRatingCF'] * self.cf_ensemble_weight)

        # Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('recRatingHybrid', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how='left',
                                                          left_on='movieId',
                                                          right_on='movieId')[['recRatingHybrid', 'movieId', 'title']]

        return recommendations_df