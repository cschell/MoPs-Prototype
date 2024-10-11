from enum import Enum

class DistanceAlgorithmEnumDB(str, Enum):
    euclidean = 'euclidean'
    cosine_similarity = 'cosine_similarity'
    lp_Distance = 'lpdistance'

