DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-100k/u.data"

MAIN_PATH = '../'

DATA_PATH = MAIN_PATH + 'data/ml-1m/ratings.dat'
MODEL_PATH = MAIN_PATH + 'models/'

# MODEL should be one of ["ml-1m_MLP", "ml-1m_GMF", "ml-1m_Neu_MF"]
MODEL = 'ml-1m_MLP'

MLP_LAYER = 3