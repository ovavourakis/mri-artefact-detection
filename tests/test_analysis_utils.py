from mri_artefact_detection.evaluation.analysis_utils import *
import pandas as pd
import numpy as np

def test_load_predictions_and_ground_truth(tmp_path):
    test_df = pd.DataFrame({'image':[1,2,3],'bin_gt':[0,0,1],'pred0':[0,0,1]}) 
    test_df.to_csv(tmp_path / 'my_test_predictions.tsv', sep='\t')

    gt, m = load_predictions_and_ground_truth(tmp_path / 'my_test_predictions.tsv')

    assert len(gt) == len(test_df)
    assert len(m) == len(test_df)

def test_assign_class():

    assert assign_class(0.6) == 1
    assert assign_class(0.3) == 0


def test_compute_mean_class_and_uncertainty():
    test_data = {
        'pred_1':  [0.6, 0.3, 0.5],
        'pred_2':  [0.7, 0.2, 0.4],
        'pred_3':  [0.8, 0.4, 0.6],
        'pred_4':  [0.5, 0.3, 0.5],
        'pred_5':  [0.6, 0.2, 0.4]
    }
    test_df = pd.DataFrame(test_data) 
    mean, std = compute_mean_class_and_uncertainty(test_df, num_mc_runs=5, thresh=0.5)
    
    # First image: 0.6, 0.7, 0.8, 0.5, 0.6 (all superior to 0.5)
    assert mean.iloc[0] == 1.0
    assert std.iloc[0] == 0.0

    # Second image: 0.3, 0.2, 0.4, 0.3, 0.2 (all inferior to 0.5)
    assert mean.iloc[1] == 0.0
    assert std.iloc[1] == 0.0

    # Third image: 0.5, 0.4, 0.6, 0.5, 0.4  (mixed)
    assert 0 < mean.iloc[2] < 1
    assert std.iloc[2] > 0 


def test_compute_mean_prob_and_uncertainty():
    test_data = {
        'pred_1':  [0.6, 0.3],
        'pred_2':  [0.7, 0.2],
        'pred_3':  [0.8, 0.4],
        'pred_4':  [0.5, 0.3],
        'pred_5':  [0.6, 0.2]
    }
    test_df = pd.DataFrame(test_data)
    mean, std = compute_mean_prob_and_uncertainty(test_df, num_mc_runs=5)
    assert mean.iloc[0] == np.mean([0.6, 0.7, 0.8, 0.5, 0.6])
    assert std.iloc[0] == np.std([0.6, 0.7, 0.8, 0.5, 0.6], ddof=1)
    assert mean.iloc[1] == np.mean([0.3, 0.2, 0.4, 0.3, 0.2])
    assert std.iloc[1] == np.std([0.3, 0.2, 0.4, 0.3, 0.2], ddof=1)


def test_merge_predictions_and_gt():
    
    index = ['sample1', 'sample2', 'sample3', 'sample4']
    # mean = pd.DataFrame({'mean': [0.7, 0.3, 0.8, 0.4]}, index=index)
    mean = pd.Series([0.7, 0.3, 0.8, 0.4], index=index)
    # std = pd.DataFrame({'std': [0.2, 0.1, 0.3, 0.0]}, index=index)
    std = pd.Series([0.2, 0.1, 0.3, 0.0], index=index)
    ground_truth = pd.DataFrame({'bin_gt': [1, 0, 1, 0]}, index=index)
    result = merge_predictions_and_gt(mean, std, ground_truth)

    assert len(result) == 4
    assert list(result.columns) == ['mean_pred', 'std_pred', 'scaled_std_pred', 'bin_gt']
    pd.testing.assert_series_equal(result['mean_pred'], mean, check_names=False)
    pd.testing.assert_series_equal(result['std_pred'], std, check_names=False)
    pd.testing.assert_series_equal(result['bin_gt'], ground_truth['bin_gt'], check_names=False)
    assert result['scaled_std_pred'].min() == 0  
    assert result['scaled_std_pred'].max() == 1  
    assert result['scaled_std_pred'].iloc[3] == 0  
    assert result['scaled_std_pred'].iloc[2] == 1 


def test_calculate_DFFMR_AP_AUROC():
    merged = pd.DataFrame({'mean_pred': [0.7, 0.3, 0.8, 0.4],   
                            'std_pred': [0.2, 0.3, 0.1, 0.0], 
                            'scaled_std_pred': [0.667, 1, 0.332, 0], 
                            'bin_gt': [1, 0, 1, 0]})
    
    # retain half the images
    DFFMR, AP, AUC = calculate_DFFMR_AP_AUROC(merged, 0.5)
    assert DFFMR == 0.5, 'DFFMR for first case is not 0.5'
    assert not np.isnan(AP), 'AP for first case is NaN'
    assert not np.isnan(AUC), 'AUROC for first case is NaN'

    # all images are flagged so none of them are retained
    DFFMR, AP, AUC = calculate_DFFMR_AP_AUROC(merged, 0.0)
    assert DFFMR == 1.0 , 'DFFMR for second case is not 1.0'
    assert np.isnan(AP), 'AP for second case is not NaN'
    assert np.isnan(AUC), 'AUROC for second case is not NaN'


def test_pointwise_metrics():
    retained_all_neg = pd.DataFrame({
        'mean_pred': [0.3, 0.4],
        'bin_gt': [0, 0]
    })
    F1, precision, recall, accuracy, specificity = pointwise_metrics(retained_all_neg, theta=0.5)
    assert np.isnan(F1)  
    assert np.isnan(precision)  
    assert np.isnan(recall)  
    assert accuracy == 1.0 
    assert specificity == 1.0 

    retained_perfect = pd.DataFrame({
        'mean_pred': [0.8, 0.2, 0.7, 0.3],
        'bin_gt': [1, 0, 1, 0]
    })
    F1, precision, recall, accuracy, specificity = pointwise_metrics(retained_perfect, theta=0.5)
    assert F1 == 1.0
    assert precision == 1.0
    assert recall == 1.0
    assert accuracy == 1.0
    assert specificity == 1.0
    
    retained_all_wrong = pd.DataFrame({
        'mean_pred': [0.2, 0.8, 0.3, 0.7],
        'bin_gt': [1, 0, 1, 0]
    })
    F1, precision, recall, accuracy, specificity = pointwise_metrics(retained_all_wrong, theta=0.5)
    assert F1 == 0.0
    assert precision == 0.0
    assert recall == 0.0
    assert accuracy == 0.0
    assert specificity == 0.0
    

    retained_mixed = pd.DataFrame({
        'mean_pred': [0.8, 0.2, 0.3, 0.7],
        'bin_gt': [1, 0, 1, 0]
    })
    F1, precision, recall, accuracy, specificity = pointwise_metrics(retained_mixed, theta=0.5)
    assert 0.0 < F1 < 1.0
    assert 0.0 < precision < 1.0
    assert 0.0 < recall < 1.0
    assert accuracy == 0.5 
    assert specificity == 0.5 
    


