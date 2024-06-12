import json
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def train_tsne_embedding(data_path):
    #task_id = int(data_path.split('.')[-2].split('_')[-1])
    task_id = 6

    with open(data_path, 'r') as f:
        data = json.load(f)
        f.close()

    y = []
    X = []
    for i in range(5):
        y = y + [f'{i}'] * len(data[i])
        X.append(np.array(data[i]))

    y = y + [f'{task_id}'] * len(data[task_id])
    X.append(np.array(data[task_id]))
    X = np.concatenate(X, axis=0)
    tsne = TSNE(n_components=2, init='pca', random_state=501, method='exact')
    X_tsne = tsne.fit_transform(X)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    df_data = {'tsne1': X_norm[:, 0], 'tsne2': X_norm[:, 1], 'task_id': y}
    df = pd.DataFrame(df_data)
    sns.scatterplot(x="tsne1", y="tsne2", hue="task_id", data=df)
    plt.savefig(f"rand_6.jpg", format="jpg")
    print('finish!')
    # plt.show()


if __name__ == '__main__':
    import glob

    files = glob.glob('./' + 'rand.json')
    for file in files:
        train_tsne_embedding(file)
