from RecommenderAlgorithm  import RecommenderAlgorithm
from RecommenderData import RecommenderData

class RecommenderComparer:
    
    def __init__(self, recommenderData, algo_comparison_set):
        self.evaluation_data = recommenderData
        self.algo_comparison_set = algo_comparison_set
        self.comparison_results = {}

    # # **kwargs options: {"sample_topN_for_userIDs": [userID_1, userID_2, ...]}
    # def Compare(self, doTopN=True, verbose=True, **kwargs):
    #     evaluated_metrics = {}
    #     for algorithm, name in self.algo_comparison_set:
    #         if(verbose):
    #             print("\nEvaluating {0} algorithm...".format(name))
    #         recommenderAlgorithm = RecommenderAlgorithm(algorithm, name)
    #         evaluated_metrics = recommenderAlgorithm.Evaluate2(self.evaluation_data, doTopN, n=10, verbose=verbose, kwargs)
    #         self.comparison_results.update(evaluated_metrics)
    #         if(verbose):
    #             print("\nAlgorithm {0} evaluation done\n".format(name))
        
    #     return self.comparison_results

    def Compare(self, doTopN=True, verbose=True, **kwargs):
        sample_topN_for_userIDs = kwargs.get("sample_topN_for_userIDs")
        evaluated_metrics = {}
        for algorithm, name in self.algo_comparison_set:
            if(verbose):
                print("\nEvaluating {0} algorithm...".format(name))
            recommenderAlgorithm = RecommenderAlgorithm(algorithm, name)
            evaluated_metrics = recommenderAlgorithm.Evaluate(self.evaluation_data, doTopN, n=10, verbose=verbose, sample_topN_for_userIDs=sample_topN_for_userIDs)
            self.comparison_results.update(evaluated_metrics)
            if(verbose):
                print("\nAlgorithm {0} evaluation done\n".format(name))
        
        return self.comparison_results
        
    # def CompareSampleTopN(self, userIDs, verbose=True):
    #     sample_TopN_all = {}
    #     sample_TopN_user = []
        
    #     for algorithm, name in self.algo_comparison_set:
    #         # if(verbose):
    #         #     print("\nEvaluating {0} algorithm...".format(name))
    #         for userID in userIDs:            
    #             recommenderAlgorithm = RecommenderAlgorithm(algorithm, name)
    #             sample_TopN_user = recommenderAlgorithm.SampleTopN(self.evaluation_data, userIDs, n=10, verbose=verbose)
    #             self.sample_TopN_all[userID] = sample_TopN_user
    #         # if(verbose):
    #         #     print("\nAlgorithm {0} evaluation done\n".format(name))
        
    #     return sample_TopN_all
