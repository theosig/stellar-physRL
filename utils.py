import numpy as np
"""
utils.py
This module contains various PyTorch-based neural network classes and utility functions 
for building and training models, particularly for spectrum encoding, decoding, and 
variational autoencoders (VAEs). The code includes implementations of multi-layer 
perceptrons (MLPs), custom activation functions, spectrum encoders/decoders, and 
loss/metric functions.
Classes:
--------
- MLP: A configurable multi-layer perceptron with support for dropout and custom activations.
- SpeculatorActivation: A custom activation function based on the Speculator paper.
- SpectrumEncoder: A CNN-based encoder with attention and MLP for latent space compression.
- SpectrumDecoder: A decoder for generating spectra from latent vectors.
- MaskedOutputLayer: A layer for combining outputs based on specific masks.
- Feedforward: A simple feedforward neural network with optional output constraints.
- FullCovarianceVAE: A full covariance variational autoencoder with a custom latent space.
Functions:
- l2_relative_error: Computes the L2 relative error between true and predicted values.
- digitize: Converts continuous values into discrete categories based on bins.
- get_correlated_prior: Generates a prior covariance matrix with specified correlations.
- kl_divergence_with_prior: Computes the KL divergence between a posterior and a prior distribution.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_conditioned = 2

class MLP(nn.Sequential):
    """Multi-Layer Perceptron

    A simple implementation with a configurable number of hidden layers and
    activation functions.

    Parameters
    ----------
    n_in: int
        Input dimension
    n_out: int
        Output dimension
    n_hidden: list of int
        Dimensions for every hidden layer
    act: list of callables
        Activation functions after every layer. Needs to have len(n_hidden) + 1
        If `None`, will be set to `LeakyReLU` for every layer.
    dropout: float
        Dropout probability
    """

    def __init__(self, n_in, n_out, n_hidden=(16, 16, 16), act=None, dropout=0):

        if act is None:
            act = [
                nn.LeakyReLU(),
            ] * (len(n_hidden) + 1)
        assert len(act) == len(n_hidden) + 1

        layer = []
        n_ = [n_in, *n_hidden, n_out]
        for i in range(len(n_) - 1):
            layer.append(nn.Linear(n_[i], n_[i + 1]))
            layer.append(act[i])
            layer.append(nn.Dropout(p=dropout))

        super(MLP, self).__init__(*layer)



class SpeculatorActivation(nn.Module):
    """Activation function from the Speculator paper

    .. math:

        a(\mathbf{x}) = \left[\boldsymbol{\gamma} + (1+e^{-\boldsymbol\beta\odot\mathbf{x}})^{-1}(1-\boldsymbol{\gamma})\right]\odot\mathbf{x}

    Paper: Alsing et al., 2020, ApJS, 249, 5

    Parameters
    ----------
    n_parameter: int
        Number of parameters for the activation function to act on
    plus_one: bool
        Whether to add 1 to the output
    """

    def __init__(self, n_parameter, plus_one=False):
        super().__init__()
        self.plus_one = plus_one
        self.beta = nn.Parameter(torch.randn(n_parameter), requires_grad=True)
        self.gamma = nn.Parameter(torch.randn(n_parameter), requires_grad=True)

    def forward(self, x):
        """Forward method

        Parameters
        ----------
        x: `torch.tensor`

        Returns
        -------
        x': `torch.tensor`, same shape as `x`
        """
        # eq 8 in Alsing+2020
        x = (self.gamma + (1 - self.gamma) * torch.sigmoid(self.beta * x)) * x
        if self.plus_one:
            return x + 1
        return x
    

class SpectrumEncoder(nn.Module):
    """Spectrum encoder

    Modified version of the encoder by Serrà et al. (2018), which combines a 3 layer CNN
    with a dot-product attention module. This encoder adds a MLP to further compress the
    attended values into a low-dimensional latent space.

    Paper: Serrà et al., https://arxiv.org/abs/1805.03908

    Parameters
    ----------
    instrument: :class:`spender.Instrument`
        Instrument that observed the data
    n_latent: int
        Dimension of latent space
    n_hidden: list of int
        Dimensions for every hidden layer of the :class:`MLP`
    act: list of callables
        Activation functions after every layer. Needs to have len(n_hidden) + 1
        If `None`, will be set to `LeakyReLU` for every layer.
    dropout: float
        Dropout probability
    """

    def __init__(
        self, n_latent=32, n_hidden=(128, 64), act=None, dropout=0
    ):

        super(SpectrumEncoder, self).__init__()
        self.n_latent = n_latent

        filters = [128, 256, 512]
        sizes = [5, 11, 21]
        self.conv1, self.conv2, self.conv3 = self._conv_blocks(
            filters, sizes, dropout=dropout
        )
        self.n_feature = filters[-1] // 2

        # pools and softmax work for spectra and weights
        self.pool1, self.pool2 = tuple(
            nn.MaxPool1d(s, padding=s // 2) for s in sizes[:2]
        )
        self.softmax = nn.Softmax(dim=-1)

        # small MLP to go from CNN features to latents
        if act is None:
            act = [nn.PReLU(n) for n in n_hidden]
            # last activation identity to have latents centered around 0
            #act.append(nn.Identity()) #removed it cause i have the mu, logvar in betavae
            act.append(nn.PReLU(n_latent))
        self.mlp = MLP(
            self.n_feature, self.n_latent, n_hidden=n_hidden, act=act, dropout=dropout
        )

    def _conv_blocks(self, filters, sizes, dropout=0):
        convs = []
        for i in range(len(filters)):
            f_in = 1 if i == 0 else filters[i - 1]
            f = filters[i]
            s = sizes[i]
            p = s // 2
            conv = nn.Conv1d(
                in_channels=f_in,
                out_channels=f,
                kernel_size=s,
                padding=p,
            )
            norm = nn.InstanceNorm1d(f)
            act = nn.PReLU(f)
            drop = nn.Dropout(p=dropout)
            convs.append(nn.Sequential(conv, norm, act, drop))
        return tuple(convs)

    def _downsample(self, x):
        # compression
        x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.conv3(x)
        C = x.shape[1] // 2
        # split half channels into attention value and key
        h, a = torch.split(x, [C, C], dim=1)

        return h, a

    def forward(self, y):
        """Forward method

        Parameters
        ----------
        y: `torch.tensor`, shape (N, L)
            Batch of observed spectra

        Returns
        -------
        s: `torch.tensor`, shape (N, n_latent)
            Batch of latents that encode `spectra`
        """
        # run through CNNs
        h, a = self._downsample(y)
        # softmax attention
        a = self.softmax(a)

        # attach hook to extract backward gradient of a scalar prediction
        # for Grad-FAM (Feature Activation Map)
        if ~self.training and a.requires_grad == True:
            a.register_hook(self._attention_hook)

        # apply attention
        x = torch.sum(h * a, dim=2)

        # run attended features into MLP for final latents
        x = self.mlp(x)
        return x

    @property
    def count_parameters(self):
        """Number of parameters in this model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _attention_hook(self, grad):
        self._attention_grad = grad

    @property
    def attention_grad(self):
        """Gradient of the attention weights

        Factor to compute the importance of attention for Grad-FAM method.

        Requires a previous `loss.backward` call for any scalar loss function based on
        outputs of this class's `forward` method. This functionality is switched off
        during training.
        """
        if hasattr(self, "_attention_grad"):
            return self._attention_grad
        else:
            return None

class SpectrumDecoder(nn.Module):
    """Spectrum decoder
    Paper: Alsing et al., 2020, ApJS, 249, 5

    Simple :class:`MLP` to create a restframe spectrum from a latent vector,
    followed by explicit redshifting, resampling, and convolution transformations to
    match the observations from a given instrument.

    Parameter
    ---------
    wave_rest: `torch.tensor`
        Restframe wavelengths
    n_latent: int
        Dimension of latent space
    n_hidden: list of int
        Dimensions for every hidden layer of the :class:`MLP`
    act: list of callables
        Activation functions after every layer. Needs to have len(n_hidden) + 1
        If `None`, will be set to :class:`SpeculatorActivation` for every layer.
    dropout: float
        Dropout probability
    """

    def __init__(
        self,
        wave_rest,
        n_latent=3,
        n_hidden=(64, 256, 1024),
        act=None,
        dropout=0,
        plus_one=True,
    ):

        super(SpectrumDecoder, self).__init__()

        if act is None:
            act = [SpeculatorActivation(n) for n in n_hidden]
            act.append(SpeculatorActivation(len(wave_rest), plus_one=plus_one))

        self.mlp = MLP(
            n_latent,
            len(wave_rest),
            n_hidden=n_hidden,
            act=act,
            dropout=dropout,
        )

        self.n_latent = n_latent

        # register wavelength tensors on the same device as the entire model
        self.register_buffer("wave_rest", wave_rest)

    def decode(self, s):
        """Decode latents into restframe spectrum

        Parameter
        ---------
        s: `torch.tensor`, shape (N, S)
            Batch of latents

        Returns
        -------
        x: `torch.tensor`, shape (N, L)
            Batch of restframe spectra
        """
        return self.mlp.forward(s)

    def forward(self, s, instrument=None, z=None):
        """Forward method

        Parameter
        ---------
        s: `torch.tensor`, shape (N, S)
            Batch of latents
        instrument: :class:`spender.Instrument`
            Instrument to generate spectrum for
        z: `torch.tensor`, shape (N, 1)
            Redshifts for each spectrum

        Returns
        -------
        y: `torch.tensor`, shape (N, L)
            Batch of spectra at redshift `z` as observed by `instrument`
        """
        # restframe
        spectrum = self.decode(s)
        return spectrum


    @property
    def count_parameters(self):
        """Number of parameters in this model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

def digitize(u, x_bins, n_cat):
    x_indices, y_indices = torch.bucketize(u, x_bins, right=False).T
    x_indices = torch.clamp(x_indices - 1, min=0)  # Ensure non-negative indices
    y_indices = torch.clamp(y_indices - 1, min=0)  # Ensure non-negative indices
    labels = x_indices * n_cat + y_indices
    return labels




class MaskedOutputLayer(nn.Module):
    def __init__(self, mask_fe, mask_a, mask_C):
        super(MaskedOutputLayer, self).__init__()
        # Store the masks
        self.mask_global=mask_a+mask_fe+mask_C
        # Convert the mask to a float tensor for multiplication
        self.mask_a = mask_a[self.mask_global] 
        self.mask_fe = mask_fe[self.mask_global]
        self.mask_carbon = mask_C[self.mask_global]

    def forward(self, decoder_fe_output, decoder_alpha_output, decoder_carbon_output):
        # Initialize an empty output tensor with the shape of the combined mask
        batch_size = decoder_fe_output.size(0)
        output = torch.zeros(batch_size, self.mask_global.sum(), dtype=decoder_fe_output.dtype).to(device)

        # Assign decoder outputs to the correct positions based on the individual masks
        output[:,self.mask_fe] += decoder_fe_output
        output[:,self.mask_carbon] += decoder_carbon_output
        output[:,self.mask_a] = decoder_alpha_output 
        return output
    


class Feedforward(nn.Module):
    def __init__(self,structure,activation=nn.ReLU(),limit_out=False,prob_out=False):
        super(Feedforward, self).__init__()

        self.layers = []
        self.limit_out=limit_out
        self.prob_out=prob_out
        for i in range(len(structure)-2):
            self.layers.append(nn.Linear(structure[i],structure[i+1]))
            self.layers.append(activation)
    
        self.layers.append(nn.Linear(structure[-2],structure[-1]))
        if self.prob_out:
            self.layers.append(nn.Softmax())
        
        self.fc = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.fc(x)
        if self.limit_out:
            x=1.-torch.relu(x)# (torch.tanh(x)+1)/2
        return x
    

# metrics & losses
def l2_relative_error(y_true,y_pred):
    return np.mean(np.linalg.norm(y_true-y_pred,axis=-1)/np.linalg.norm(y_true,axis=-1))

class FullCovarianceVAE(nn.Module):
    def __init__(self,encoder,decoder_fe, decoder_alpha, decoder_carbon, mask_fe, 
                          mask_a, 
                          mask_C,
                          latent_space_size = 6):
        super(FullCovarianceVAE, self).__init__()

        self.encoder = encoder
        self.decoder_fe = decoder_fe
        self.decoder_alpha = decoder_alpha
        self.decoder_carbon = decoder_carbon
        self.regressor = Feedforward([n_conditioned, n_conditioned]).to(device)
        self.type = 'bVAE_regressor'
        self.latent_space_size = latent_space_size
        self.output_layer = MaskedOutputLayer(torch.tensor(mask_fe).to(device),torch.tensor(mask_a).to(device),torch.tensor(mask_C).to(device))
        self.mean = nn.Linear(32,latent_space_size)
        self.fc_ltril = nn.Linear(32, latent_space_size * (latent_space_size + 1) // 2) 
        #linear layer for conditioning part
        self.conditioning = nn.Linear(32, 2)

    def reparameterize(self, mean, L):
        """
        Sample z from the multivariate Gaussian: z = mean + L * eps,
        where eps ~ N(0, I).
        """
        batch_size = mean.size(0)
        eps = torch.randn(batch_size, self.latent_space_size, device=mean.device)
        # Use batch matrix multiplication to compute L * eps for each sample.
        z = mean + torch.bmm(L, eps.unsqueeze(2)).squeeze(2)
        return z

    
    def count_parameters(self):
        return np.sum([np.prod(x.size()) for x in self.parameters()])

    def encode(self, x):
        h = self.encoder(x)
        #Latent Space
        mu = self.mean(h)
        l_params = self.fc_ltril(h)  # Shape: [batch_size, latent_dim*(latent_dim+1)//2]

        batch_size = x.size(0)
        
        # Initialize L as zeros: [batch_size, latent_space_size, latent_space_size]
        L = torch.zeros(batch_size, self.latent_space_size, self.latent_space_size, device=x.device)
        
        # Get indices for the lower-triangular part (including the diagonal)
        tril_indices = torch.tril_indices(row=self.latent_space_size, col=self.latent_space_size, offset=0)
        # Fill L with the parameters from l_params
        L[:, tril_indices[0], tril_indices[1]] = l_params

        # Ensure the diagonal elements are positive for a valid Cholesky factor.
        diag_indices = torch.arange(self.latent_space_size, device=x.device)
        L[:, diag_indices, diag_indices] = F.softplus(L[:, diag_indices, diag_indices])

        z = self.reparameterize(mu, L)
        x = self.conditioning(h)

        return z, x, mu, L


    def forward(self, x, alpha_activation=1.):
        z, x, mu, L = self.encode(x)
        
        x_fe = torch.cat((z[:,-3:-2],z[:,-1:].detach(), x),1)        
        x_alpha=torch.cat((z[:,-3:-2].detach(), z[:,-1:].detach(), z[:,-2:-1]*alpha_activation,x),1)
        x_carbon=torch.cat((z[:,-3:-2].detach(),z[:,-1:],x),1)

        x_alpha = self.decoder_alpha(x_alpha)
        x_fe = self.decoder_fe(x_fe)
        x_carbon = self.decoder_carbon(x_carbon)
        tefflogg = self.regressor(x)

        output=self.output_layer(x_fe, x_alpha, x_carbon)
        return output, z, mu, L, tefflogg


    def predict(self, dataloader, alpha_activation=1.):
        predictions_reconstructed = []
        predictions_encoded = []
        mu_all = []
        logvar_all = []
        atmpar_all = []
        
        self.eval() # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            for batch in dataloader:
                X_batch, err_batch = batch
                X_batch=X_batch.to(device)
                reconstructed, encoded, mu, logvar, atmpar = self(X_batch, alpha_activation)
                
                # Append batch outputs to the respective lists
                predictions_reconstructed.append(reconstructed)
                predictions_encoded.append(encoded)
                mu_all.append(mu)
                logvar_all.append(logvar)
                atmpar_all.append(atmpar)

        
        # Concatenate predictions from all batches
        predictions_reconstructed = torch.cat(predictions_reconstructed, dim=0)
        predictions_encoded = torch.cat(predictions_encoded, dim=0)
        mu_all = torch.cat(mu_all, dim=0)
        logvar_all = torch.cat(logvar_all, dim=0)
        atmpar_all = torch.cat(atmpar_all, dim=0)
        
        return predictions_reconstructed, predictions_encoded, mu_all,logvar_all, atmpar_all


def get_correlated_prior(rhoFe_alpha=-.3, rho_FE_C=-.17, latent_dim=3, device="cpu"):
    """
    Returns the prior covariance matrix, its inverse, and its log-determinant.
    Assumes correlation only between the first and second latent dimensions.
    """
    # Create the covariance matrix
    prior_cov = torch.eye(latent_dim, device=device)
    if latent_dim >= 2:
        prior_cov[0, 1] = rhoFe_alpha
        prior_cov[1, 0] = rhoFe_alpha

        prior_cov[0, 2] = rho_FE_C
        prior_cov[2, 0] = rho_FE_C

    # Compute inverse and log determinant
    prior_cov_inv = torch.inverse(prior_cov)
    prior_log_det = torch.logdet(prior_cov)
    return prior_cov, prior_cov_inv, prior_log_det
_, prior_cov_inv, prior_log_det = get_correlated_prior(device=device)





def kl_divergence_with_prior(mean, L, prior_cov_inv, prior_log_det):
    """
    Compute the KL divergence between:
      q(z|x) = N(mean, Sigma_q) with Sigma_q = L L^T, and
      p(z) = N(0, Sigma_prior) with a given full covariance.
    
    KL divergence is given by:
      KL(q||p) = 0.5 * [ log(det(Sigma_prior)/det(Sigma_q))
                          - k + trace(prior_cov_inv * Sigma_q) + mu^T prior_cov_inv mu ]
    
    Args:
      mean: Tensor of shape [batch_size, latent_dim]
      L: Tensor of shape [batch_size, latent_dim, latent_dim] (Cholesky factors for Sigma_q)
      prior_cov_inv: Inverse of the prior covariance matrix (latent_dim x latent_dim)
      prior_log_det: Log determinant of the prior covariance matrix (scalar)
      
    Returns:
      kl: Tensor of shape [batch_size] with the KL divergence for each sample.
    """
    batch_size, latent_dim = mean.size()
    
    # Compute log(det(Sigma_q)) = 2 * sum(log(diag(L)))
    diag_L = torch.diagonal(L, dim1=1, dim2=2)
    log_det_Sigma_q = 2 * torch.log(diag_L).sum(dim=1)
    
    # Compute Sigma_q = L L^T for each sample
    Sigma_q = torch.bmm(L, L.transpose(1, 2))  # shape: (batch_size, latent_dim, latent_dim)
    
    # Compute trace(prior_cov_inv * Sigma_q) for each sample.
    # Using Einstein summation: "ij,bji->b" multiplies prior_cov_inv (ij) with Sigma_q (bji) and sums appropriately.
    trace_term = torch.einsum("ij,bji->b", prior_cov_inv, Sigma_q)
    
    # Compute quadratic term: mu^T prior_cov_inv mu for each sample
    mu_term = torch.einsum("bi,ij,bj->b", mean, prior_cov_inv, mean)
    
    # KL divergence for each sample
    kl = 0.5 * (prior_log_det - log_det_Sigma_q - latent_dim + trace_term + mu_term)
    
    return kl.mean()


