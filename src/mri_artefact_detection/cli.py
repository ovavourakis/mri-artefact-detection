import click
from mri_artefact_detection.training.train import train_model
from mri_artefact_detection.inference.inference import run_model_inference
from mri_artefact_detection.evaluation.run_analysis import evaluate

@click.group()
def main():
    pass

@main.command()
@click.option('--savedir', required=True, type=click.Path(), help='Directory where training outputs and checkpoints will be saved. (default: required)')
@click.option('--datadir', required=True, type=click.Path(), help='Directory containing the dataset. (default: required)')
@click.option('--datasets', default='artefacts1,artefacts2,artefacts3', help='Comma-separated list of dataset names to be used for training. (default: "artefacts1,artefacts2,artefacts3")')
@click.option('--contrasts', default='T1wMPR', help='Comma-separated list of MRI contrasts to be considered. (default: "T1wMPR")')
@click.option('--quals', default='clean,exp_artefacts', help='Comma-separated list of quality labels (e.g., "clean", "exp_artefacts"). (default: "clean,exp_artefacts")')
@click.option('--random-affine', default=1/12, type=float, help='Distribution weight for RandomAffine artefact. (default: 0.0833)')
@click.option('--random-elastic-deformation', default=1/12, type=float, help='Distribution weight for RandomElasticDeformation artefact. (default: 0.0833)')
@click.option('--random-anisotropy', default=1/12, type=float, help='Distribution weight for RandomAnisotropy artefact. (default: 0.0833)')
@click.option('--rescale-intensity', default=1/12, type=float, help='Distribution weight for RescaleIntensity artefact. (default: 0.0833)')
@click.option('--random-motion', default=1/12, type=float, help='Distribution weight for RandomMotion artefact. (default: 0.0833)')
@click.option('--random-ghosting', default=1/12, type=float, help='Distribution weight for RandomGhosting artefact. (default: 0.0833)')
@click.option('--random-spike', default=1/12, type=float, help='Distribution weight for RandomSpike artefact. (default: 0.0833)')
@click.option('--random-bias-field', default=1/12, type=float, help='Distribution weight for RandomBiasField artefact. (default: 0.0833)')
@click.option('--random-blur', default=1/12, type=float, help='Distribution weight for RandomBlur artefact. (default: 0.0833)')
@click.option('--random-noise', default=1/12, type=float, help='Distribution weight for RandomNoise artefact. (default: 0.0833)')
@click.option('--random-swap', default=1/12, type=float, help='Distribution weight for RandomSwap artefact. (default: 0.0833)')
@click.option('--random-gamma', default=1/12, type=float, help='Distribution weight for RandomGamma artefact. (default: 0.0833)')
@click.option('--target-clean-ratio', default=0.5, type=float, help='Fraction of clean images to be resampled in the training set. (default: 0.5)')
@click.option('--mc-runs', default=20, type=int, help='Number of Monte Carlo runs on the test set. (default: 20)')
def train(savedir, datadir, datasets, contrasts, quals, random_affine, random_elastic_deformation, random_anisotropy, 
          rescale_intensity, random_motion, random_ghosting, random_spike, random_bias_field, random_blur, 
          random_noise, random_swap, random_gamma, target_clean_ratio, mc_runs):
    """
    Command to train the MRI artefact detection model.
    """
    datasets = tuple(datasets.split(','))
    contrasts = tuple(contrasts.split(','))
    quals = tuple(quals.split(','))
    
    train_model(
        savedir=savedir,
        datadir=datadir,
        datasets=datasets,
        contrasts=contrasts,
        quals=quals,
        random_affine=random_affine,
        random_elastic_deformation=random_elastic_deformation,
        random_anisotropy=random_anisotropy,
        rescale_intensity=rescale_intensity,
        random_motion=random_motion,
        random_ghosting=random_ghosting,
        random_spike=random_spike,
        random_bias_field=random_bias_field,
        random_blur=random_blur,
        random_noise=random_noise,
        random_swap=random_swap,
        random_gamma=random_gamma,
        target_clean_ratio=target_clean_ratio,
        mc_runs=mc_runs
    )

@main.command()
@click.option('--savedir', required=True, type=click.Path(), help='Directory where inference outputs will be saved.')
@click.option('--weights', required=True, type=click.Path(), help='Path to the pre-trained model weights.')
@click.option('--gt-data', required=True, type=click.Path(), help='Path to the ground truth data file.')
@click.option('--mc-runs', default=20, type=int, help='Number of Monte Carlo runs on the test set. (default: 20)')
def infer(savedir, weights, gt_data, mc_runs):
    """
    Command to perform inference using the MRI artefact detection model.
    """
    run_model_inference(
        savedir=savedir,
        weights=weights,
        gt_data=gt_data,
        mc_runs=mc_runs
    )

@main.command()
@click.option('--model-preds', required=True, type=click.Path(), help='Path to the model predictions file.')
@click.option('--ternary', is_flag=True, help='Flag to indicate ternary analysis. (default: False)')
def eval(model_preds, ternary):
    """
    Command to evaluate the MRI artefact detection model.
    """
    evaluate(
        MODEL_PREDS=model_preds,
        TERNARY=ternary
    )