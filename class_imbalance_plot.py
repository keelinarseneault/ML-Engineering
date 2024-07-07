from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as n
import os

def plot_scatter_tsne(X, y, classes, labels, colors, markers, loc, dir_name, fig_name, random_seed):
    """
    Plot the scatter plot using TSNE
    
    Parameters
    ----------
    X : the feature matrix
    y : the target vector
    classes : the classes in the target vector
    labels : the labels for different classes
    colors : the colors for different classes
    markers : the markers for different classes
    loc : the location of the legend
    dir_name : the name of the directory
    fig_name : the name of the figure
    random_seed : the random seed
    """
    
    # Make directory
    directory = os.path.dirname(dir_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the tsne transformed training feature matrix
    X_embedded = TSNE(n_components=2, random_state=random_seed).fit_transform(X)

    # Get the tsne dataframe
    tsne_df = pd.DataFrame(np.column_stack((X_embedded, y)), columns=['x1', 'x2', 'y'])

    # Get the data
    data = {}
    for class_ in classes:
        data_x1 = [tsne_df['x1'][i] for i in range(len(tsne_df['y'])) if tsne_df['y'][i] == class_]
        data_x2 = [tsne_df['x2'][i] for i in range(len(tsne_df['y'])) if tsne_df['y'][i] == class_]
        data[class_] = [data_x1, data_x2]
    
    # The scatter plot
    fig = plt.figure(figsize=(8, 6))
    
    for class_, label, color, marker in zip(classes, labels, colors, markers):
        data_x1, data_x2 = data[class_]
        plt.scatter(data_x1, data_x2, c=color, marker=marker, s=120, label=label)

    # Set x-axis
    plt.xlabel('x1')

    # Set y-axis
    plt.ylabel('x2')

    # Set legend
    plt.legend(loc=loc)

    # Save and show the figure
    plt.tight_layout()
    plt.savefig(dir_name + fig_name)
    plt.show()