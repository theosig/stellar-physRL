import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def createplot(ax, true, pred, size="20%", pad=0):
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size=size, pad=pad)
    ax.figure.add_axes(ax2)
    ax.plot(true)
    ax.plot(pred)
    ax2.axhline(0.,c="k",ls="dashed")
    ax2.plot(true-pred, color="crimson")
    ax.set_xticks([])




def plot_correlations(y,mu):
    correlations_teff = np.array([np.corrcoef(mu[:, i], y[:,0])[0, 1] for i in range(mu.shape[1])])
    correlations_logg = np.array([np.corrcoef(mu[:, i], y[:,1])[0, 1] for i in range(mu.shape[1])])

    correlations_feh = np.array([np.corrcoef(mu[:, i], y[:,-3])[0, 1] for i in range(mu.shape[1])])
    correlations_a = np.array([np.corrcoef(mu[:, i], y[:,-2])[0, 1] for i in range(mu.shape[1])])
    correlations_C = np.array([np.corrcoef(mu[:, i], y[:,-1])[0, 1] for i in range(mu.shape[1])])

    # Prepare data for seaborn plot
    data = {
        'Feature': np.tile(range(mu.shape[1]), 5),  # Repeat feature indices 5 times (for each label)
        'Correlation': np.concatenate([correlations_teff, correlations_logg, correlations_feh, correlations_a, correlations_C]),
        'Label': ['Teff'] * mu.shape[1] + ['logg'] * mu.shape[1] + ['Fe/H'] * mu.shape[1] + ['Alpha'] * mu.shape[1] + ['Carbon'] * mu.shape[1]
    }

    # Convert to a pandas DataFrame for seaborn
    df = pd.DataFrame(data)


    # Create a grouped bar plot using seaborn
    fig_bar=plt.figure(figsize=(12, 6))
    ax=sns.barplot(x='Feature', y='Correlation', hue='Label', data=df,palette="magma")
    for i in range(mu.shape[1]):
        if i % 2 == 0:  # Alternate shading for clarity
            plt.axvspan(i - 0.5, i + 0.5, color='royalblue', alpha=0.1)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='--')

    if mu.shape[1]==5:
        plt.xlim(-.5,4.5)
        plt.xticks(np.arange(mu.shape[1]),labels=[
                                        r"$z_1$",
                                        r"$z_2$",
                                        r"$z_{\rm [Fe/H]}$",
                                        r"$z_{\rm [\alpha/H]}$",
                                        r"$z_{\rm [C/H]}$"],fontsize=15)
    elif mu.shape[1]==3:
        plt.xlim(-.5,2.5)
        plt.ylim(-1.,1.)
        plt.xticks(np.arange(mu.shape[1]),labels=[
                                    r"$z_{\rm [Fe/H]}$",
                                    r"$z_{\rm [\alpha/H]}$",
                                    r"$z_{\rm [C/H]}$"],fontsize=15)


    plt.title('Correlation of Encoded Features with Different Labels')
    plt.xlabel('')
    plt.ylabel('Correlation')
    plt.legend(title='Labels')



    fig, ax = plt.subplots(1,3,figsize=(19,5))
    fig1=ax[1].scatter(y[:,-2], mu[:,-2], alpha=1, s=10, c=y[:,-3]) 

    ax[1].set_ylabel("enc_alpha");ax[1].set_xlabel(r"[$\alpha$/Fe]")
    plt.colorbar(fig1,label=r"[Fe/H]")


    figC=ax[2].scatter(y[:,-1], mu[:,-1], alpha=1,s=10,c=y[:,-3])

    ax[2].set_ylabel("enc_carbon");ax[2].set_xlabel(r"[C/Fe]")
    plt.colorbar(figC,label=r"[Fe/H]")


    fig2=ax[0].scatter(y[:,-3], mu[:,-3], alpha=1,s=10,c=y[:,-1])
    ax[0].set_ylabel("enc_fe");ax[0].set_xlabel("[Fe/H]")
    plt.colorbar(fig2,label=r"[$\alpha$/Fe]")

    ax[1].axvline(x=0,c="orange")
    ax[0].axvline(x=-1,c="orange")
    ax[2].axvline(x=.7,c="orange")
    return fig_bar, fig


