from Recommender_import_list import *


HYPERPARAMETERS = {
    TopPop: {
        'movielens1m_ours': {},
        'amazon_instant_video': {}
    },
    UserKNNCFRecommender: {
        'movielens1m_ours': {
            'cosine': {
                'topK': 418,
                'shrink': 402,
                'similarity': 'cosine',
                'normalize': True,
                'feature_weighting': 'TF-IDF',
            },
        },
        'amazon_instant_video': {
            'cosine': {
                'topK': 800,
                'shrink': 364,
                'similarity': 'cosine',
                'normalize': False,
                'feature_weighting': 'TF-IDF',
            },
        },
    },
    ItemKNNCFRecommender: {
        'movielens1m_ours': {
            'cosine': {
                'topK': 197,
                'shrink': 0,
                'similarity': 'cosine',
                'normalize': True,
                'feature_weighting': 'TF-IDF',
            },
        },
        'amazon_instant_video': {
            'cosine': {
                'topK': 998,
                'shrink': 21,
                'similarity': 'cosine',
                'normalize': False,
                'feature_weighting': 'TF-IDF',
            },
        },
    },
    P3alphaRecommender: {
        'movielens1m_ours': {
            'topK': 350, 'alpha': 0.6537, 'normalize_similarity': True
        },
        'amazon_instant_video': {
            'topK': 1000, 'alpha': 0.3705, 'normalize_similarity': False
        },
    },
    RP3betaRecommender: {
        'movielens1m_ours': {
            'topK': 853,
            'alpha': 0.0000,
            'beta': 0.4098,
            'normalize_similarity': True,
        },
        'amazon_instant_video': {
            'topK': 442,
            'alpha': 0.6540,
            'beta': 0.0332,
            'normalize_similarity': False,
        },
    },
    
    SLIMElasticNetRecommender: {
        'movielens1m_ours': {
            'topK': 642,
            'l1_ratio': 1.89e-5,
            'alpha': 0.0490,
        },
        'amazon_instant_video': {
            'topK': 862,
            'l1_ratio': 6.11e-5,
            'alpha': 0.5507,
        },
    },
}
