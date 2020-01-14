import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.api as sm
import operator

def print_corr(df, pct=0):
    
    sns.set(style="white")

    # Compute the correlation matrix
    if pct == 0:
        corr = df.corr()
    else:
        corr = abs(df.corr()) > pct

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

def p_dropping(df, target, p_value):

    features = list(df.drop([target], axis=1).columns)
    drop_from_x = [target]
    # Fitting the actual model
    y = df[target].astype('float')
    X = sm.add_constant(df.drop([target], axis=1).astype('float'))
    model = sm.OLS(y, X)
    results = model.fit()
    dict_p_values = results.pvalues.to_dict()
    del dict_p_values["const"]
    max_p_param = max(dict_p_values.items(), key=operator.itemgetter(1))[0]
    max_p = max(dict_p_values.values())
    best_model = results
    while max_p > p_value:
        #Fitting the new model
        drop_from_x.append(max_p_param)
        y = df[target].astype('float')
        X = sm.add_constant(df.drop(drop_from_x, axis=1).astype('float'))
        model = sm.OLS(y, X)
        results = model.fit()
        dict_p_values = results.pvalues.to_dict()
        del dict_p_values["const"]
        max_p_param = max(dict_p_values.items(), key=operator.itemgetter(1))[0]
        max_p = max(dict_p_values.values())
        best_model = results
    return best_model, drop_from_x