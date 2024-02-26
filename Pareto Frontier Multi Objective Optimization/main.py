from pareto.ObjectiveSpace import *
import pandas as pd
import numpy as np
import os

def run_scenario(dataset, scenario, objectives, reference_point):
    dir = os.listdir(f'data/{dataset}')
    results = []
    for element in dir:
        model = pd.read_csv(f'data/{dataset}/{element}', sep='\t')
        obj = ObjectivesSpace(model, objectives)
        print('****** OPTIMAL *****')
        non_dominated = obj.get_nondominated()
        print(non_dominated)

        # Save non-dominated results based on scenario
        if len(objectives) == 3:  # scenario1
            non_dominated_file = f'results/{dataset}/{element[:-4]}_{list(objectives.keys())[0]}_{list(objectives.keys())[1]}_{list(objectives.keys())[2]}_not_dominated.tsv'
        else:  # scenario2
            non_dominated_file = f'results/{dataset}/{element[:-4]}_{list(objectives.keys())[0]}_{list(objectives.keys())[1]}_not_dominated.tsv'
        non_dominated.to_csv(non_dominated_file, sep='\t', index=False)

        print('****** DOMINATED *****')
        print(obj.get_dominated())
        obj.plot(obj.get_nondominated(), obj.get_dominated(), reference_point)

        # Calculate metrics
        ms = obj.maximum_spread()
        sp = obj.spacing()
        er = obj.error_ratio()
        hv = obj.hypervolumes(reference_point)
        c = non_dominated.shape[0]
        hv_c = hv / c
        print(ms, sp, er, hv, c, hv_c)

        results.append([element[:-4], ms, sp, er, hv, c, hv_c])

    # Save metrics results based on scenario
    res_df = pd.DataFrame(results, columns=['model', 'MS', 'SP', 'ER', 'HV', 'C', 'HV/C'])
    metrics_file = f'results/{dataset}/{scenario}_metrics_results.csv'
    res_df.to_csv(metrics_file, index=False)

if __name__ == '__main__':
    dataset = 'movielens1m'
    
    # Scenario 1
    scenario1_objectives = {'nDCG': 'max', 'Gini': 'max', 'EPC': 'max'}
    scenario1_reference_point = np.array([0, 0, 0])
    run_scenario(dataset, 'scenario1', scenario1_objectives, scenario1_reference_point)
    
    # Scenario 2
    scenario2_objectives = {'nDCG': 'max', 'APLT': 'max'}
    scenario2_reference_point = np.array([0, 0])
    run_scenario(dataset, 'scenario2', scenario2_objectives, scenario2_reference_point)
