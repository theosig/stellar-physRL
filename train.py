
"""
# Stellar Physics Representation Learning (PhysRL) Training Script 
This script implements a training pipeline for a Variational Autoencoder (VAE) model 
designed to extract chemical abundances directly from data, with no predetermined chemical labels. 
## Credits:
- Author: Theosamuele Signor
- Institution: Universidad Diego Portales/ Inria Chile
## References:
- Signor et al. 2025, aa55376-25. Towards model-free stellar chemical abundances
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd
from xgboost import XGBRegressor
from utils import *
from plot_utils import plot_correlations, createplot
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
alpha_regions = pd.read_csv("lineregions/GPT7_regions_alpha.csv",sep=",") #these have have lee2011
fe_regions = pd.read_csv("lineregions/inverted_regions_Fe.csv",sep=",")
carbon_regions = pd.read_csv("lineregions/GPT_regions_carbon.csv",sep=",")



from torch.utils.tensorboard import SummaryWriter
log_dir = "runs/train"
writer = SummaryWriter(log_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# %%
print("Reading data...")
data_dir = "/home/tsignor/Documents/RLdata/"
flux = np.load(data_dir+"fluxes.npy", mmap_mode='r')
errors = np.load(data_dir+"errors.npy", mmap_mode='r')

metadata = np.load(data_dir+'metadata.npy') #Teff, logg, FeH, aFe, CFe
wl = np.load(data_dir+'wavelegths.npy')
Y=metadata #Teff, logg, FeH, aFe, CFe

print(flux.shape, metadata.shape, wl.shape)



# %% Filtering regions
mask_a = np.zeros_like(wl, dtype=bool)
mask_fe = np.zeros_like(wl, dtype=bool)
mask_C = np.zeros_like(wl, dtype=bool)

# Vectorized approach to update the mask
for index, row in alpha_regions.iterrows():
    mask_a = mask_a + ((wl >= row['wave_base']) & (wl <= row['wave_top']))

for index, row in fe_regions.iterrows():
    mask_fe = mask_fe + ((wl >= row['wave_base']) & (wl <= row['wave_top']))
    
for index, row in carbon_regions.iterrows():
    mask_C = mask_C+ ((wl >= row['wave_base']) & (wl <= row['wave_top']))

print("alpha mask length", mask_a.sum()/len(mask_a))
print("Iron mask length", mask_fe.sum()/len(mask_fe))
print("Carbon mask length", mask_C.sum()/len(mask_C))

overlap_mask = np.logical_and(mask_a, mask_fe+mask_C)
print("alpha overlap: ",overlap_mask.sum()/mask_a.sum())

overlap_mask = np.logical_and(mask_C, mask_fe+mask_a)
print("C-Fe overlap: ",overlap_mask.sum()/mask_C.sum())

# Remove overlaps
mask_fe_without_overlap = mask_fe & ~mask_a
mask_C_without_overlap = mask_C & ~mask_a 
mask_C=mask_C_without_overlap
mask_fe=mask_fe_without_overlap

data_mask=torch.tensor(mask_a+mask_fe+mask_C).to(device)



# %%
from sklearn.model_selection import train_test_split
indices = np.arange(flux.shape[0])
train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=0)
X_train, X_test = flux[train_indices], flux[test_indices]
y_train, y_test = Y[train_indices], Y[test_indices]

noise_std = errors[train_indices]
noise_std_test = errors[test_indices]


# %%
# hyperparameters
hparams_config = {
    "n_cat": 8, 
    "n_z": 3,
    "n_bins": X_train.shape[1],
}
initial_metrics = {
    "Reconstruction Loss": np.nan,  # Placeholder value
    "Regularization Loss": np.nan,  # Placeholder value
    "KL Divergence Loss": np.nan    # Placeholder value
}

# Log Hyperparameters at the Start
writer.add_hparams(hparams_config, initial_metrics)

n_z=hparams_config["n_z"]
n_conditioned = 2


n_bins=hparams_config["n_bins"]
n_cat = hparams_config["n_cat"]

LR = 1e-3
num_epochs = 5000
# %%

encoder=SpectrumEncoder()
decoder_alpha = SpectrumDecoder(n_hidden=(32, 64, 128), n_latent=3+n_conditioned, wave_rest=torch.tensor(wl[mask_a]))
decoder_fe = SpectrumDecoder(n_hidden=(32, 64, 128), n_latent=2+n_conditioned,wave_rest=torch.tensor(wl[mask_fe]))
decoder_carbon = SpectrumDecoder(n_hidden=(32, 64, 128), n_latent=2+n_conditioned,wave_rest=torch.tensor(wl[mask_C]))

discriminador = Feedforward([n_z, 64, 128, n_cat**n_conditioned],activation=nn.ReLU(),prob_out=True).to(device)
model = FullCovarianceVAE(encoder, decoder_fe,decoder_alpha,decoder_carbon, 
                          mask_fe, 
                          mask_a, 
                          mask_C,n_z).to(device)
print(model.count_parameters())

# %%
optimizer_ae = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-8)
optimizer_reg = torch.optim.Adam(discriminador.parameters(), lr=LR, eps=1e-8)

batch_size = 2**9
n_batches=int(np.ceil(len(X_train)/batch_size))

scheduler_ae = torch.optim.lr_scheduler.OneCycleLR(optimizer_ae, LR,
                                              epochs=num_epochs, steps_per_epoch=n_batches)
scheduler_reg = torch.optim.lr_scheduler.OneCycleLR(optimizer_reg, LR,
                                              epochs=num_epochs, steps_per_epoch=n_batches)

# %%

sc=StandardScaler()
y_sc=sc.fit_transform(y_train)
y_sc_test=sc.transform(y_test) 

# %%

x_bins = torch.linspace(y_sc[:, 0].min(), y_sc[:, 0].max(), n_cat).to(device)
y_bins = torch.linspace(y_sc[:, 1].min(), y_sc[:, 1].max(), n_cat).to(device)

def log_lik_normal(x_true, x_pred, data_mask, sigma = 1.0):
    log_var = torch.log(sigma[:,data_mask]**2)
    loss= -0.5*(torch.log(torch.tensor(2*torch.pi)) + torch.mean(log_var + (x_pred - x_true[:,data_mask]) ** 2 * torch.exp(-log_var), axis=-1))
    return -torch.mean(loss)

criterion_reg = nn.CrossEntropyLoss()
criterion_rec = log_lik_normal

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(noise_std, dtype=torch.float32), torch.tensor(y_sc, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_loader_val = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

val_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(noise_std_test, dtype=torch.float32), torch.tensor(y_sc_test, dtype=torch.float32))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# %%


epoch=1


wl_tensor=torch.tensor(wl).to(device)
# %%
def train_step(train_loader, lambda_rec, lambda_clf, lambda_kl,alpha_activation=1):
    train_reg_loss, train_rec_loss, train_kl_loss, train_tefflogg_loss = 0.0, 0.0, 0.0, 0.0
    
    for inputs, errors, targets in train_loader:
        x, x_err, u = inputs.to(device), errors.to(device), targets.to(device)

        optimizer_ae.zero_grad()
        # Forward pass through the autoencoder
        decoded, encoded, mu, log_var, predicted_ureg = model(x, alpha_activation)

        # Discriminator step
        optimizer_reg.zero_grad()
        predicted_u = discriminador(mu.detach())  # Detach to train the discriminator only
        labels = digitize(u[:, :2], x_bins, n_cat).to(device).long()
        err_reg_disc = criterion_reg(predicted_u, labels)  # Discriminator loss
        err_reg_disc.backward()  # Backprop for discriminator
        optimizer_reg.step()  # Update discriminator weights


        # Autoencoder step
        predicted_u = discriminador(mu)  # Full gradient flow for autoencoder training
        err_rec = criterion_rec(x_true=x, 
                                x_pred=decoded, 
                                    data_mask=data_mask,
                                sigma=x_err,
                                )  # Reconstruction loss
        
        err_kl = kl_divergence_with_prior(mu, log_var, prior_cov_inv, prior_log_det) # KL divergence loss
        err_reg = criterion_reg(predicted_u, labels)  # Classification loss for AE
        err_tefflogg = nn.MSELoss()(predicted_ureg, u[:,:2])   

        # Total loss for autoencoder
        err_tot = lambda_rec * err_rec - lambda_clf * err_reg + lambda_kl * err_kl + err_tefflogg 
        err_tot.backward()  # Backprop for autoencoder
        optimizer_ae.step()  # Update autoencoder weights
        #end new
        scheduler_ae.step()  
        scheduler_reg.step()

        train_reg_loss += err_reg.cpu().detach().numpy()
        train_rec_loss += err_rec.cpu().detach().numpy()
        train_kl_loss += err_kl.cpu().detach().numpy()
        train_tefflogg_loss += err_tefflogg.cpu().detach().numpy()

    train_reg_loss /= len(train_loader)
    train_rec_loss /= len(train_loader)
    train_kl_loss /= len(train_loader)
    train_tefflogg_loss /= len(train_loader)

    return train_reg_loss, train_rec_loss, train_kl_loss, train_tefflogg_loss


lambda_rec=1; lambda_kl=1e-3
X_t = torch.tensor(X_test, dtype = torch.float32).to(device)   
y_t = torch.tensor(y_sc_test, dtype = torch.float32).to(device)  

# %%
alpha_activation = 1.
for epoch in tqdm(range(epoch, num_epochs)):    
    if epoch <= 200:
        lambda_clf = 0
    elif 200 < epoch <= 2000:
        lambda_clf = 1e-2
    elif 2000 < epoch:
        lambda_clf = 1e-1
    if epoch > 500:
        alpha_activation=1.

    discriminador.train()  # Set the model to training mode
    model.train()  # Set the model to training mode

    train_reg_loss,train_rec_loss,train_kl_loss,train_tefflogg_loss=train_step(train_loader,lambda_rec,lambda_clf,lambda_kl,alpha_activation=alpha_activation)

    # ...log the loss...
    writer.add_scalar('rec train loss',
                       train_rec_loss,
                       epoch)
    writer.add_scalar('clf train loss',
                       train_reg_loss,
                       epoch)
    writer.add_scalar('KL train loss',
                       train_kl_loss,
                       epoch)
    writer.add_scalar('Teff-logg train loss',
                          train_tefflogg_loss,
                          epoch)



    discriminador.eval()  # Set the model to evaluation mode
    model.eval()


    val_reg_loss, val_rec_loss, val_kl_loss,val_tefflogg_loss,  val_corr_fe, val_corr_a, val_corr_C = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0

    with torch.no_grad():  # No need to track gradients during validation
        for val_inputs,val_errors, val_targets in val_loader:
            val_x, val_err, val_u = val_inputs.to(device), val_errors.to(device), val_targets.to(device)

            # Forward pass
            val_decoded,val_encoded,mu,log_var,val_predicted_tefflogg = model(val_x,alpha_activation)

            #val_pred_hist = kernel_density_estimate(val_encoded[:,:2], grid, bandwidth)
            val_predicted_u = discriminador(mu)
            # Compute validation losses
            val_err_rec = criterion_rec(x_true=val_x, 
                                        x_pred=val_decoded, 
                                        data_mask=data_mask,
                                        sigma=val_err) # Reconstruction loss

            labels_test = digitize(val_u[:, :2],x_bins,n_cat).to(device).long()
            val_err_reg = criterion_reg(val_predicted_u, labels_test)
            val_err_kl = kl_divergence_with_prior(mu, log_var, prior_cov_inv, prior_log_det)
            val_err_tefflogg = nn.MSELoss()(val_predicted_tefflogg, val_u[:,:2])

            corr_coeffs = torch.corrcoef(torch.cat([mu[:, -3:], val_u[:,-3:]],dim=1).T).cpu().detach().numpy() 


            # Extract the correlation coefficients between corresponding features:
            # Index (0, 3) for the first feature of `matrix1` with the first feature of `matrix2`,
            # Index (1, 4) for the second feature with the second, etc.           
            correlations_feh = corr_coeffs[0, 0+3]
            correlations_a = corr_coeffs[1, 1+3]
            correlations_C = corr_coeffs[2, 2+3]

            val_corr_fe += correlations_feh
            val_corr_a += correlations_a
            val_corr_C += correlations_C

            val_reg_loss += val_err_reg.cpu().detach().numpy()
            val_rec_loss += val_err_rec.cpu().detach().numpy()
            val_kl_loss += val_err_kl.cpu().detach().numpy()
            val_tefflogg_loss += val_err_tefflogg.cpu().detach().numpy()

    # Normalize validation losses by the number of validation batches
    val_reg_loss /= len(val_loader)
    val_rec_loss /= len(val_loader)
    val_kl_loss /= len(val_loader)
    val_tefflogg_loss /= len(val_loader)

    val_corr_fe /= len(val_loader)
    val_corr_a /= len(val_loader)
    val_corr_C /= len(val_loader)


    writer.add_scalar('rec val loss',
                       val_rec_loss,
                       epoch)
    writer.add_scalar('clf val loss',
                       val_reg_loss,
                       epoch)
    writer.add_scalar('KL val loss',
                       val_kl_loss,
                       epoch)
    writer.add_scalar('Teff-logg val loss',
                            val_tefflogg_loss,
                            epoch)
    

    writer.add_scalar('Corr FE',
                       abs(val_corr_fe),
                       epoch)
    writer.add_scalar('Corr alpha',
                       abs(val_corr_a),
                       epoch)
    writer.add_scalar('Corr C',
                       abs(val_corr_C),
                       epoch)
    
    

    if not (epoch-1)%25:
        torch.save(model.state_dict(), "checkpoints/AE_"+log_dir.split('runs/')[1])
        torch.save(discriminador.state_dict(), "checkpoints/discriminador_"+log_dir.split('runs/')[1])

        xg = XGBRegressor()

        reconstruced, encoded, mu_train, atmpar_train = model.predict(train_loader_val,alpha_activation)
        encoded = encoded.cpu().detach().numpy()
        mu_train = mu_train.cpu().detach().numpy()
        reconstruced,encoded_test,mu_test, atmpar_test = model.predict(val_loader,alpha_activation) 
        encoded_test=encoded_test.cpu().detach().numpy()
        mu_test = mu_test.cpu().detach().numpy()
        atmpar_test = atmpar_test.cpu().detach().numpy()

        xg.fit(mu_train,y_train[:,:2])
        pred = xg.predict(mu_test)

        writer.add_scalar('xgb clf l2',
                       l2_relative_error(pred, y_test[:,:2]),
                       epoch)
        writer.add_scalar('reconstructed l2',
                       l2_relative_error(X_test[:,data_mask.cpu().detach()], reconstruced.cpu().detach().numpy()),
                       epoch)

        fig_bar, fig_scatter = plot_correlations(y_test, mu_test) 
        writer.add_figure('Correlation Bar',
                            fig_bar,
                            global_step=epoch)
        
        writer.add_figure('Correlation scatter',
                            fig_scatter,
                            global_step=epoch)

        fig_rec,ax=plt.subplots(1,1,figsize=(10,5))
        createplot(ax, X_test[0,data_mask.cpu().detach()], reconstruced.cpu().detach().numpy()[0],size="30%")
        writer.add_figure('reconstruction',
                            fig_rec,
                            global_step=epoch)


        #create a figure with scatterplot atmpar_test vs y_test[:,:2] residuals
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].scatter(y_sc_test[:, 0], atmpar_test[:, 0], s=1)
        ax[0].set_xlabel('Teff'); ax[0].set_ylabel('Teff pred')
        ax[1].scatter(y_sc_test[:, 1], atmpar_test[:, 1], s=1)
        ax[1].set_xlabel('logg'); ax[1].set_ylabel('logg pred')
        writer.add_figure('tefflogg',
                          fig,
                          global_step=epoch)


        
