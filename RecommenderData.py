from MovieLens import MovieLens
from surprise.model_selection import LeaveOneOut
from surprise.model_selection import train_test_split
from surprise import KNNBaseline
import pandas as pd
# from itertools import chain

class RecommenderData:

    def __init__(self, ratingsFilePath, moviesFilePath, verbose=True):
        self.ratingsPath = ratingsFilePath
        self.moviesPath = moviesFilePath
        
        if(verbose):
                print("\nLoading Movies and Ratings...")

        # load data
        self.movielens = MovieLens(self.ratingsPath, self.moviesPath)
        self.ratings = self.movielens.loadMovieLensLatestSmall()
        self.popularity_rankings = self.movielens.getPopularityRanks()
        
        ## Section for creating dataset for using full-input-dataset for training/test
        
        self.trainset_full = self.ratings.build_full_trainset()
        
        # create antitest set from full training set
        self.antitestset_full = self.trainset_full.build_anti_testset()

        ## Section for creating dataset for using train-test-split for training/test        
    
        # 75/25 train/test split 
        self.trainset_percent_split, self.testset_percent_split = train_test_split(self.ratings,test_size=0.25, random_state=1, shuffle=True)     


        # ## Section for creating dataset for using leave-one-out method for training/cv/test 

        #Build a "leave one out" train/test split for evaluating top-N recommenders
        LOOCV = LeaveOneOut(n_splits=1, random_state=1)
        for loocv_train, loocv_test in LOOCV.split(self.ratings):
            self.trainset_loocv = loocv_train 
            self.testset_loocv = loocv_test
            self.antitestset_loocv = self.trainset_loocv.build_anti_testset()  

        ## Compute similarty matrix between items so we can measure diversity
        similarity_options = {'name': 'cosine', 'user_based': False}
        self.similarity_algorithm = KNNBaseline(sim_options=similarity_options)
        self.similarity_algorithm.fit(self.trainset_full)

        if(verbose):
                print("\nMovies and Ratings loaded\n")

    def GetFullTrainSet(self):
        return self.trainset_full
    
    def GetFullAntiTestSet(self):
        return self.antitestset_full
    
    def GetAntiTestSetForUser(self, testSubject):
        trainset = self.trainset_full
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(str(testSubject))
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                 i in trainset.all_items() if
                                 i not in user_items]
        return anti_testset

    def GetTrainSet(self):
        return self.trainset_percent_split
    
    def GetTestSet(self):
        return self.testset_percent_split
    
    def GetLOOCVTrainSet(self):
        return self.trainset_loocv
    
    def GetLOOCVTestSet(self):
        return self.testset_loocv
    
    def GetLOOCVAntiTestSet(self):
        return self.antitestset_loocv
    
    def GetSimilarities(self):
        return self.similarity_algorithm
    
    def GetPopularityRankings(self):
        return self.popularity_rankings

    def GetMovieName(self, movieID):
        return self.movielens.getMovieName(movieID)

    # take UserID and return the top N movies (by ratings) rated by the user
    def GetTopNRatedByUser(self, userID, n=10):
        userRatings = self.movielens.getUserRatings(userID)
        movie_list = []
        df = pd.DataFrame(userRatings, columns=["movieID", "rating"])
        df_sorted = df.sort_values(by=['rating'], inplace=False, ascending=False)
        row_count = 1
        for indx, row in df_sorted.iterrows():
            # get movie name
           movie_list.append(self.GetMovieName(int(row[0])))
           row_count += 1
           if row_count > n:
               break
        return movie_list