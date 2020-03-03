from RecommenderData import RecommenderData
from RecommenderMetrics import RecommenderMetrics

class RecommenderAlgorithm:
    def __init__(self, algorithm, name):
        self.recommender_algorithm = algorithm
        self.recommender_name = name

    # def Evaluate(self, evaluationData, doTopN, n=10, verbose=True):
    #     self.evaluated_metrics = {}
    #     self.N = n
    #     self.recommender_data = evaluationData

    #     # Use train-test-split dataset for RMSE and MAE scores
    #     self.recommender_algorithm.fit(self.recommender_data.GetTrainSet())
    #     predictions = self.recommender_algorithm.test(self.recommender_data.GetTestSet())
    #     self.evaluated_metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
    #     self.evaluated_metrics["MAE"] = RecommenderMetrics.MAE(predictions)

    #     # do only if we want top N recommendations. Compute intensive operation
    #     if(doTopN):
    #         # use Leave one out algorithm
    #         self.recommender_algorithm.fit(self.recommender_data.GetLOOCVTrainSet())
    #         leftout_predictions = self.recommender_algorithm.test(self.recommender_data.GetLOOCVTestSet())
    #         predictions_all_minus_train = self.recommender_algorithm.test(self.recommender_data.GetLOOCVAntiTestSet())
    #         topN_predictions = RecommenderMetrics.GetTopN(predictions_all_minus_train, self.N, minimumRating=4.0)

    #         self.evaluated_metrics["HitRate"] = RecommenderMetrics.HitRate(topN_predictions, leftout_predictions) 
    #         self.evaluated_metrics["CumulativeHitRate"] = RecommenderMetrics.CumulativeHitRate(topN_predictions, leftout_predictions) 
    #         # self.evaluated_metrics["RatingHitRate"] = RecommenderMetrics.RatingHitRate(topN_predictions, leftout_predictions) 
    #         self.evaluated_metrics["AverageReciprocalHitRank"] = RecommenderMetrics.AverageReciprocalHitRank(topN_predictions, leftout_predictions)

    #     # use full dataset for these metrics: UserCoverage, Diversity, Novelty
    #     trainset_full = self.recommender_data.GetFullTrainSet()
    #     self.recommender_algorithm.fit(trainset_full)
    #     predictions_all = self.recommender_algorithm.test(self.recommender_data.GetFullAntiTestSet())
    #     topN_predictions = RecommenderMetrics.GetTopN(predictions_all, self.N)
    #     self.evaluated_metrics["UserCoverage"] = RecommenderMetrics.UserCoverage(topN_predictions, trainset_full.n_users, ratingThreshold=4.0)
        
    #     # Diversity uses the similarity matrix
    #     self.evaluated_metrics["Diversity"] = RecommenderMetrics.Diversity(topN_predictions, self.recommender_data.GetSimilarities())
        
    #     # Novelty uses the Popularity rankings
    #     self.evaluated_metrics["Novelty"] = RecommenderMetrics.Novelty(topN_predictions, self.recommender_data.GetPopularityRankings())

    #     # format: {Algorithm: {evaluated metrics}}
    #     return {self.recommender_name: self.evaluated_metrics}

    def FilterTopN(self, predictions, userIDs):
        sample_TopN_all = {}
        # sample_TopN_userspecific = []

        for userID in userIDs:
            #for movie_count in range(n):
                # for movieID, estimated_ratings in predictions[userID]
            #     print("prediction {0} for userID {1}: {2}".format(movie_count, userID, predictions[int(userID)]))
            sample_TopN_all[userID] = [self.recommender_data.GetMovieName(movieID) for movieID, rating in predictions[int(userID)]]

        return sample_TopN_all

    # kwargs options: {sample_topN_for_userID: }
    def Evaluate(self, evaluationData, doTopN, n=10, verbose=True, sample_topN_for_userIDs=[]):
        sample_topN = {}
        self.evaluated_metrics = {}
        self.N = n
        output = {}

        self.recommender_data = evaluationData

        # sample_TopN_all = {}
        # sample_TopN_user = []

        # # creating dictionary like: userID -> algorithms -> Top N
        # for userID in sample_topN_for_userIDs:
        #     algos = {}
        #     for algorithm, name in self.algo_comparison_set:
        #         algos.update({name: [])
        #     sample_TopN_all[userID] = algos

        # Use train-test-split dataset for RMSE and MAE scores
        self.recommender_algorithm.fit(self.recommender_data.GetTrainSet())
        predictions = self.recommender_algorithm.test(self.recommender_data.GetTestSet())
        self.evaluated_metrics["RMSE"] = RecommenderMetrics.RMSE(predictions)
        self.evaluated_metrics["MAE"] = RecommenderMetrics.MAE(predictions)

        # do only if we want top N recommendations. Compute intensive operation
        if(doTopN):
            # use Leave one out algorithm
            self.recommender_algorithm.fit(self.recommender_data.GetLOOCVTrainSet())
            leftout_predictions = self.recommender_algorithm.test(self.recommender_data.GetLOOCVTestSet())
            predictions_all_minus_train = self.recommender_algorithm.test(self.recommender_data.GetLOOCVAntiTestSet())
            topN_predictions = RecommenderMetrics.GetTopN(predictions_all_minus_train, self.N, minimumRating=4.0)

            self.evaluated_metrics["HitRate"] = RecommenderMetrics.HitRate(topN_predictions, leftout_predictions) 
            self.evaluated_metrics["CumulativeHitRate"] = RecommenderMetrics.CumulativeHitRate(topN_predictions, leftout_predictions) 
            # self.evaluated_metrics["RatingHitRate"] = RecommenderMetrics.RatingHitRate(topN_predictions, leftout_predictions) 
            self.evaluated_metrics["AverageReciprocalHitRank"] = RecommenderMetrics.AverageReciprocalHitRank(topN_predictions, leftout_predictions)

        # use full dataset for these metrics: UserCoverage, Diversity, Novelty
        trainset_full = self.recommender_data.GetFullTrainSet()
        self.recommender_algorithm.fit(trainset_full)
        predictions_all = self.recommender_algorithm.test(self.recommender_data.GetFullAntiTestSet())
        topN_predictions = RecommenderMetrics.GetTopN(predictions_all, self.N)

        if len(sample_topN_for_userIDs) != 0:
            sample_topN = self.FilterTopN(topN_predictions, sample_topN_for_userIDs)

        self.evaluated_metrics["UserCoverage"] = RecommenderMetrics.UserCoverage(topN_predictions, trainset_full.n_users, ratingThreshold=4.0)
        
        # Diversity uses the similarity matrix
        self.evaluated_metrics["Diversity"] = RecommenderMetrics.Diversity(topN_predictions, self.recommender_data.GetSimilarities())
        
        # Novelty uses the Popularity rankings
        self.evaluated_metrics["Novelty"] = RecommenderMetrics.Novelty(topN_predictions, self.recommender_data.GetPopularityRankings())

        # format: {Algorithm: {evaluated metrics}}
        output[self.recommender_name] = {  "metrics" : self.evaluated_metrics }

        if len(sample_topN_for_userIDs) != 0:
             # format: {TopN: {userID: {Algorithm: [Sample_TopN movies recommended]}}}
            output[self.recommender_name].update({ "sample_topn" : sample_topN})
        
        return output

