from mri_artefact_detection.evaluation.analysis_utils import *
import pandas as pd

def test_load_predictions_and_ground_truth(tmp_path):
    test_df = pd.DataFrame({'image':[1,2,3],'bin_gt':[0,0,1],'pred0':[0,0,1]}) 
    test_df.to_csv(tmp_path / 'my_test_predictions.tsv', sep='\t')

    gt, m = load_predictions_and_ground_truth(tmp_path / 'my_test_predictions.tsv')

    assert len(gt) == len(test_df)
    assert len(m) == len(test_df)

