from RecommenderData import RecommenderData
from RecommenderComparer import RecommenderComparer
from surprise import SVD, SVDpp
from surprise import NormalPredictor
import numpy as np
import random
from tabulate import tabulate
import pandas as pd

ratingsPath = 'ratings.csv'
moviesPath = 'movies.csv'
Test_userIDs = ["85", "91"]
doTopN = True

# seed for reproducibility
np.random.seed(0)
random.seed(0)

# for expanded display in pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# initialize the algorithms before comparison
SVD_Algorithm = SVD(random_state=10)
# SVDpp_Algorithm = SVDpp(random_state=10)
Normal_Predictor = NormalPredictor()

# creating the comparison set
algo_comparison_set = [(SVD_Algorithm, "SVD"), (Normal_Predictor, "Normal")]

# set data
recommenderData = RecommenderData(ratingsPath, moviesPath, verbose=True)



# set comparer
recommenderComparer = RecommenderComparer(recommenderData, algo_comparison_set)
# compare
comparison = recommenderComparer.Compare(doTopN, verbose=True, sample_topN_for_userIDs=Test_userIDs) 
# comparison["0000"] = {"sample_topn": }



# # tabulating the comparison
header = ["Algorithm"]
rows = []
is_header_done = False

# for algo, all_metrics in comparison.items():
#     row = [algo]
#     for metric_type, metric_value in all_metrics.items():
#         row.extend([metric_value])
#         # create header only once
#         if is_header_done == False:
#             header.extend([metric_type])
#     is_header_done = True
#     rows.extend([row])

metrics = {}
for algo in comparison.keys():
    row = [algo]
    metrics = comparison[algo]["metrics"]
    for metric_type, metric_value in metrics.items():
        row.extend([metric_value])
        # create header only once
        if is_header_done == False:
            header.extend([metric_type])
    is_header_done = True
    rows.extend([row])

print("\nComparison results:\n")
print(tabulate(rows, headers=header))
print("\nLegend:\n")
print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
print("MAE:       Mean Absolute Error. Lower values mean better accuracy.")
if (doTopN):
    print("HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.")
    print("cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better.")
    print("ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better." )
    print("Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better.")
    print("Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations")
    print("           for a given user. Higher means more diverse.")
    print("Novelty:   Average popularity rank of recommended items. Higher means more novel.")

if len(Test_userIDs) > 0:
    user_orig_ratings = {}
    rating_set = {}
    for userID in Test_userIDs:
        rating_set[userID] = recommenderData.GetTopNRatedByUser(int(userID), n=10)
    comparison["AAAAOriginal"] = {"sample_topn": rating_set}

    print(comparison)

    header_topn = ["UserID", "Algorithm"]
    row_topn = []
    table_topn = []
    is_header_topn_done = False
    movie_headers = []

    for algo in comparison.keys():
        topn = comparison[algo]["sample_topn"]
        for userID, predictions in topn.items():
            if is_header_topn_done == False:
                for i in range(len(predictions)):
                    h = "Movie " + str(i + 1)
                    movie_headers.append(h)
                is_header_topn_done = True
            row_topn = [userID]
            row_topn.append(algo)
            row_topn.extend(predictions)
            table_topn.append(row_topn)
    header_topn.extend(movie_headers)
    df = pd.DataFrame(table_topn)
    df.columns = header_topn
    df.sort_values(by=['UserID'], inplace=True)
    print(df.T)
    df.T.to_csv(r'/Users/udasatap/Documents/AI ML DS/RecSys-Materials/U_Framework/data/recommendations.csv')
