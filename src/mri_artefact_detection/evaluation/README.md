# Analysing Results

> **Advice:** First, have a look at `theory.pdf` to understand how we frame model performance evaluation.

The functions in `analysis_utils.py` provide a scaffold for an analysis of model performance along the lines of `theory.pdf`. This uses the model to classify the data into three categories: 'definitely clean', 'definitely artefact', and 'manual review needed'. 

To run this analysis on the model outputs from a testset inference run (as produced by the script in the `inference` directory) use

```
python run_ternary_analysis.py
```

You can also run a simpler analysis based on a binary classification ('clean' vs. 'artefact') using

```
python run_analysis.py
```
