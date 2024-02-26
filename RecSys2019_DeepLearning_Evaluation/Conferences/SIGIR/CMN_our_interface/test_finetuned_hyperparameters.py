from Recommender_import_list import *


HYPERPARAMETERS = {
    TopPop: {'citeulike': {}},
    UserKNNCFRecommender: {
        'citeulike': {
            'cosine': {
                'topK': 578,
                'shrink': 0,
                'similarity': 'cosine',
                'normalize': True,
                'feature_weighting': 'BM25',
            }
        },
    },
    ItemKNNCFRecommender: {
        'citeulike': {
            'cosine': {
                'topK': 594,
                'shrink': 999,
                'similarity': 'cosine',
                'normalize': True,
                'feature_weighting': 'TF-IDF',
            }
        },
    },
    P3alphaRecommender: {
        'citeulike': {
            'topK': 653, 'alpha': 0.6310, 'normalize_similarity': False
        }
    },
    RP3betaRecommender: {
        'citeulike': {
            'topK': 764,
            'alpha': 0.7110,
            'beta': 0.2297,
            'normalize_similarity': True,
        }
    },
    SLIMElasticNetRecommender: {
        'citeulike': {
            'topK': 1000,
            'l1_ratio': 4.21e-5,
            'alpha': 0.0265,
        }
    },
}
