#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

import os
import traceback
import argparse
from functools import partial

import numpy as np

from Conferences.SIGIR.CMN_our_interface.CMN_RecommenderWrapper import (
    CMN_RecommenderWrapper
)
from Conferences.SIGIR.CMN_our_interface.finetuned_hyperparameters import (
    HYPERPARAMETERS
)
from Recommender_import_list import UserKNNCFRecommender, ItemKNNCFRecommender
from ParameterTuning.run_parameter_search import (
    runParameterSearch_Collaborative
)
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from Base.Evaluation.Evaluator import EvaluatorNegativeItemSample
from Utils.ResultFolderLoader import (
    ResultFolderLoader, generate_latex_hyperparameters
)
from Utils.assertions_on_data_for_experiments import (
    assert_implicit_data, assert_disjoint_matrices
)
from Utils.plot_popularity import (
    plot_popularity_bias, save_popularity_statistics
)
from Conferences.SIGIR.CMN_our_interface.CiteULike.CiteULikeReader import (
    CiteULikeReader
)
from Conferences.SIGIR.CMN_our_interface.Pinterest.PinterestICCVReader import (
    PinterestICCVReader
)
from Conferences.SIGIR.CMN_our_interface.Epinions.EpinionsReader import (
    EpinionsReader
)


ALGORITHM_NAME = 'CMN'
CONFERENCE_NAME = 'SIGIR'
KNN_SIMILARITY_TO_REPORT_LIST = [
    'cosine', 'dice', 'jaccard', 'asymmetric', 'tversky'
]

def evaluate_recommender(
    recommender_class,
    fit_params,
    evaluator_validation,
    evaluator_test,
    URM_train,
    URM_train_last,
    output_file_name_root=None,
    **kwargs,
):
    """
    Helper function to evaluate an instance of `recommender_class` using
    `fit_params` hyperparameters.

    :param recommender_class: Class of the recommender object to optimize, it must be a BaseRecommender type.
    :param fit_params: Dictionary with hyperparameters. Will be passed to `recommender_class.fit()` call.
    :param evaluator_validation: Evaluator object to be used for the validation.
    :param evaluator_test: Evaluator object to be used for the test results.
    :param URM_train: URM matrix that will be used to fit the `recommender_class`.
    :param URM_train_last: URM matrix that will be used for the last fit.
    :param output_file_name_root: Root of the output file name. If none, `recommender_class.RECOMMENDER_NAME` will be used.
    :param kwargs: Other arguments to pass to the `SearchSingleCase().search()` call.
    """
    try:
        if output_file_name_root is None:
            output_file_name_root = recommender_class.RECOMMENDER_NAME

        parameterSearch = SearchSingleCase(
            recommender_class, evaluator_validation, evaluator_test
        )
        recommender_input_args = SearchInputRecommenderArgs([URM_train.copy()])
        recommender_input_args_last_test = SearchInputRecommenderArgs(
            [URM_train_last]
        )
        parameterSearch.search(
            recommender_input_args,
            recommender_input_args_last_test=recommender_input_args_last_test,
            fit_hyperparameters_values=fit_params,
            output_file_name_root=output_file_name_root,
            **kwargs,
        )
    except Exception as e:
        print(
            'On recommender {} Exception {}'.format(recommender_class, str(e))
        )
        traceback.print_exc()


def run_search_or_eval(
    dataset_name,
    flag_baselines_tune=False,
    flag_baselines_eval=False,
    flag_DL_article_default=False,
    flag_print_results=True,
):
    assert not (flag_baselines_tune == True and flag_baselines_eval == True),\
        'Only one of {flag_baselines_tune, flag_baselines_eval} parameters can be equal to True.'

    result_folder_path = 'result_experiments/{}/{}_{}/'.format(
        CONFERENCE_NAME, ALGORITHM_NAME, dataset_name
    )
    os.makedirs(result_folder_path, exist_ok=True)

    if dataset_name == 'citeulike':
        dataset = CiteULikeReader(result_folder_path)
    elif dataset_name == 'epinions':
        dataset = EpinionsReader(result_folder_path)
    elif dataset_name == 'pinterest':
        dataset = PinterestICCVReader(result_folder_path)
    else:
        raise RuntimeError(f'Unsupported dataset: {dataset_name}')

    URM_train = dataset.URM_DICT['URM_train'].copy()
    URM_validation = dataset.URM_DICT['URM_validation'].copy()
    URM_test = dataset.URM_DICT['URM_test'].copy()
    URM_test_negative = dataset.URM_DICT['URM_test_negative'].copy()

    # Ensure IMPLICIT data and DISJOINT sets
    assert_implicit_data(
        [URM_train, URM_validation, URM_test, URM_test_negative]
    )
    if dataset_name == 'citeulike':
        assert_disjoint_matrices([URM_train, URM_validation, URM_test])
        assert_disjoint_matrices([URM_test, URM_test_negative])
    elif dataset_name == 'pinterest':
        assert_disjoint_matrices([URM_train, URM_validation, URM_test])
        assert_disjoint_matrices([URM_train, URM_validation, URM_test_negative])
    else:
        assert_disjoint_matrices(
            [URM_train, URM_validation, URM_test, URM_test_negative]
        )

    algorithm_dataset_string = '{}_{}_'.format(ALGORITHM_NAME, dataset_name)
    plot_popularity_bias(
        [URM_train + URM_validation, URM_test],
        ['Training data', 'Test data'],
        result_folder_path + algorithm_dataset_string + 'popularity_plot',
    )
    save_popularity_statistics(
        [
            URM_train + URM_validation + URM_test,
            URM_train + URM_validation,
            URM_test,
        ],
        ['Full data', 'Training data', 'Test data'],
        result_folder_path + algorithm_dataset_string + 'popularity_statistics'
    )
    evaluator_validation = EvaluatorNegativeItemSample(
        URM_validation, URM_test_negative, cutoff_list=[5]
    )
    evaluator_test = EvaluatorNegativeItemSample(
        URM_test, URM_test_negative, cutoff_list=[5, 10]
    )

    metric_to_optimize = 'HIT_RATE'
    if flag_baselines_tune:
        runParameterSearch_Collaborative_partial = partial(
            runParameterSearch_Collaborative,
            URM_train=URM_train,
            URM_train_last_test=URM_train + URM_validation,
            metric_to_optimize=metric_to_optimize,
            evaluator_validation_earlystopping=evaluator_validation,
            evaluator_validation=evaluator_validation,
            evaluator_test=evaluator_test,
            output_folder_path=result_folder_path,
            parallelizeKNN=False,
            allow_weighting=True,
            resume_from_saved=True,
            n_cases=50,
            n_random_starts=15,
        )

        for recommender_class in HYPERPARAMETERS:
            try:
                runParameterSearch_Collaborative_partial(recommender_class)
            except Exception as e:
                print(
                    'On recommender {} Exception {}'.format(
                        recommender_class, str(e)
                    )
                )
                traceback.print_exc()
    elif flag_baselines_eval:
        evaluate_recommender_partial = partial(
            evaluate_recommender,
            evaluator_validation=evaluator_validation,
            evaluator_test=evaluator_test,
            URM_train=URM_train,
            URM_train_last=URM_train + URM_validation,
            output_folder_path=result_folder_path,
            resume_from_saved=True,
            save_model='best',
            evaluate_on_test='best',
        )
        for recommender_class in HYPERPARAMETERS:
            fit_params = HYPERPARAMETERS[recommender_class]
            if recommender_class in {
                UserKNNCFRecommender, ItemKNNCFRecommender
            }:
                for similarity_type, params in fit_params[dataset_name].items():
                    evaluate_recommender_partial(
                        recommender_class,
                        params,
                        output_file_name_root=recommender_class.RECOMMENDER_NAME + '_' + similarity_type,
                    )
            else:
                evaluate_recommender_partial(
                    recommender_class, fit_params[dataset_name]
                )

    if flag_DL_article_default:
        try:
            CMN_article_hyperparameters = {
                'epochs_gmf': 100,
                'hops': 3,
                'neg_samples': 4,
                'reg_l2_cmn': 1e-1,
                'reg_l2_gmf': 1e-4,
                'pretrain': True,
                'learning_rate': 1e-3,
                'verbose': False,
            }
            if dataset_name == 'citeulike':
                CMN_article_hyperparameters['epochs'] = 50
                CMN_article_hyperparameters['batch_size'] = 128
                CMN_article_hyperparameters['embed_size'] = 50
            elif dataset_name == 'epinions':
                CMN_article_hyperparameters['epochs'] = 45
                CMN_article_hyperparameters['batch_size'] = 128
                CMN_article_hyperparameters['embed_size'] = 40
            else:
                CMN_article_hyperparameters['epochs'] = 5
                CMN_article_hyperparameters['batch_size'] = 256
                CMN_article_hyperparameters['embed_size'] = 50

            CMN_earlystopping_hyperparameters = {
                'validation_every_n': 5,
                'stop_on_validation': True,
                'evaluator_object': evaluator_validation,
                'lower_validations_allowed': 5,
                'validation_metric': metric_to_optimize
            }

            parameterSearch = SearchSingleCase(
                CMN_RecommenderWrapper,
                evaluator_validation=evaluator_validation,
                evaluator_test=evaluator_test
            )
            recommender_input_args = SearchInputRecommenderArgs(
                CONSTRUCTOR_POSITIONAL_ARGS = [URM_train],
                FIT_KEYWORD_ARGS = CMN_earlystopping_hyperparameters
            )
            recommender_input_args_last_test = recommender_input_args.copy()
            recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train + URM_validation

            parameterSearch.search(
                recommender_input_args,
                recommender_input_args_last_test=recommender_input_args_last_test,
                fit_hyperparameters_values=CMN_article_hyperparameters,
                output_folder_path=result_folder_path,
                resume_from_saved=True,
                output_file_name_root=CMN_RecommenderWrapper.RECOMMENDER_NAME
            )
        except Exception as e:
            print(
                'On recommender {} Exception {}'.format(
                    CMN_RecommenderWrapper, str(e)
                )
            )
            traceback.print_exc()

    if flag_print_results:
        n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)
        file_name = '{}..//{}_{}_'.format(
            result_folder_path, ALGORITHM_NAME, dataset_name
        )
    
        result_loader = ResultFolderLoader(
            result_folder_path,
            base_algorithm_list=None,
            other_algorithm_list=[CMN_RecommenderWrapper],
            KNN_similarity_list=KNN_SIMILARITY_TO_REPORT_LIST,
            ICM_names_list=None,
            UCM_names_list=None,
        )
        result_loader.generate_latex_results(
            file_name + '{}_latex_results.txt'.format('article_metrics'),
            metrics_list = [
                'HIT_RATE', 'NDCG', 'UNEXPECTEDNESS', 'SERENDIPITY'
            ],
            cutoffs_list = [5, 10],
            table_title = None,
            highlight_best = True,
        )
        result_loader.generate_latex_results(
            file_name + '{}_latex_results.txt'.format('all_metrics'),
            metrics_list = [
                'PRECISION',
                'RECALL',
                'MAP_MIN_DEN',
                'MRR',
                'NDCG',
                'F1',
                'HIT_RATE',
                'ARHR_ALL_HITS',
                'NOVELTY',
                'DIVERSITY_MEAN_INTER_LIST',
                'DIVERSITY_HERFINDAHL',
                'COVERAGE_ITEM',
                'DIVERSITY_GINI',
                'SHANNON_ENTROPY',
                'UNEXPECTEDNESS',
                'SERENDIPITY',
            ],
            cutoffs_list = [10],
            table_title = None,
            highlight_best = True,
        )
        result_loader.generate_latex_time_statistics(
            file_name + '{}_latex_results.txt'.format('time'),
            n_evaluation_users=n_test_users,
            table_title=None,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mut_group = parser.add_mutually_exclusive_group()
    mut_group.add_argument(
        '-b',
        '--baseline_tune',
        help='Baseline hyperparameter search',
        type=bool,
        default=False,
    )
    mut_group.add_argument(
        '-e',
        '--baseline_eval',
        help='Only evaluate baselines using already tuned hyperparameters',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '-a',
        '--DL_article_default',
        help='Train the DL model with article hyperparameters',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '-p',
        '--print_results',
        help='Print results',
        type=bool,
        default=True,
    )

    input_flags = parser.parse_args()
    print(input_flags)

    dataset_list = ['citeulike', 'pinterest', 'epinions']
    for dataset_name in dataset_list:
        run_search_or_eval(
            dataset_name,
            input_flags.baseline_tune,
            input_flags.baseline_eval,
            input_flags.DL_article_default,
            input_flags.print_results,
        )

    if input_flags.print_results:
        generate_latex_hyperparameters(
            result_folder_path='result_experiments/{}/'.format(CONFERENCE_NAME),
            algorithm_name=ALGORITHM_NAME,
            experiment_subfolder_list=dataset_list,
            other_algorithm_list=[CMN_RecommenderWrapper],
            KNN_similarity_to_report_list=KNN_SIMILARITY_TO_REPORT_LIST,
            split_per_algorithm_type=True,
        )
