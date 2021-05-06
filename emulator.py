from recommender_system import KNNModel
from preprocessing import Preprocessor


def model_preprocessing():
    """
    Preprocessing for kNN model
    Data cached to data folder (preprocess train and test data for validated==True and full dataframe for False)
    """
    p = Preprocessor(validate=False, random_state=0, idf_normalize=False)
    p.cache_data()


if __name__ == "__main__":
    users_list = []
    while True:
        line = input()
        if line:
            users_list.append(int(line))
        else:
            break
    ub = KNNModel(validate=False, random_state=0)
    ub.fit()
    recommendations = ub.predict(users_list=users_list)
    for rec in recommendations:
        print(" ".join([str(x) for x in rec]))
