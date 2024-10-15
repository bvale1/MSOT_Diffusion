import argparse
import wandb
import logging 
import torch
import os
import json
import random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from pytorch_utils import *
from lightning_modules.lightning_VAE import LightningVAE
from lightning_modules.lightning_VAE import LightningAutoEncoder
from lightning_modules.lightning_VAE import Encoder
from lightning_modules.lightning_VAE import Decoder


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/home/wv00017/MSOT_Diffusion/ImageNet_MSOT_Dataset/', help='path to the root directory of the dataset')
    parser.add_argument('--git_hash', type=str, default='None', help='optional, git hash of the current commit for reproducibility')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_test_example', help='disable save test examples to wandb', action='store_false')
    parser.add_argument('--wandb_log', help='disable wandb logging', action='store_false', default=True)
    parser.add_argument('--latent_dim', type=int, default=1024, help='dimension of the latent space')
    parser.add_argument('--kl_weight', type=float, default=1.0, help='kl divergence weight')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--seed', type=int, default=None, help='seed for reproducibility')
    parser.add_argument('--save_model_dir', type=str, default=None, help='path to save the model')
    parser.add_argument('--save_embeddings', action='store_true', help='save latent space embeddings of the dataset')

    args = parser.parse_args()
    logging.info(f'args dict: {vars(args)}')

    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(False)
    logging.info(f'cuDNN deterministic: {torch.torch.backends.cudnn.deterministic}')
    logging.info(f'cuDNN benchmark: {torch.torch.backends.cudnn.benchmark}')
    
    if args.seed:
        seed = args.seed
    else:
        seed = np.random.randint(0, 2**32 - 1)
    logging.info(f'seed: {seed}')
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    np.random.seed(seed)
    random.seed(seed) 
    pl.seed_everything(seed, workers=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'using device: {device}')
    
    with open(os.path.join(args.root_dir, 'config.json'), 'r') as f:
        config = json.load(f) # <- dataset config contains normalisation parameters
    
    normalise_x = MeanStdNormalise(
        torch.Tensor([config['normalisation_X']['mean']]),
        torch.Tensor([config['normalisation_X']['std']])
    )
    x_transform = transforms.Compose([
        ReplaceNaNWithZero(), 
        normalise_x
    ])
    
    normalise_y = MeanStdNormalise(
        torch.Tensor([config['normalisation_mu_a']['mean']]),
        torch.Tensor([config['normalisation_mu_a']['std']])
    )
    y_transform = transforms.Compose([
        ReplaceNaNWithZero(),
        normalise_y
    ])
    
    if args.wandb_log:
        wandb.login()
        wandb_log = WandbLogger(
            project='mu_a_reconstruction', name='AutoEncoder', save_code=True, reinit=True
        )
    else:
        wandb_log = None
    
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, mode='min',
    )
    
    trainer = pl.Trainer(
        log_every_n_steps=1, check_val_every_n_epoch=1, accelerator='gpu',
        devices=1, max_epochs=args.epochs, deterministic=True, logger=wandb_log,
        callbacks=[early_stop_callback]
    )
    
    dataset = ReconstructAbsorbtionDataset(
        args.root_dir, gt_type='mu_a', X_transform=x_transform, Y_transform=y_transform
    )
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(42) # train/val/test sets are always the same
    )
    logging.info(f'train: {len(train_dataset)}, val: {len(val_dataset)}, test: \
        {len(test_dataset)}')
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=20
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=20
    )
    
    sample_size = (dataset.__getitem__(0)[0].shape[1], dataset.__getitem__(0)[0].shape[2])
    model = LightningAutoEncoder(
        # encoder (1,256,256) -> (32,128,128) -> (64,64,64) -> (128,32,32) -> (256,16,16) -> (512,8,8) -> (512,4,4) -> (1024)
        Encoder(input_size=sample_size,
                input_channels=1,
                hidden_channels=[32, 64, 128, 256, 512, 512], 
                latent_dim=args.latent_dim),
        # decoder (1024) -> (512,4,4) -> (512,8,8) -> (256,16,16) -> (128,32,32) -> (64,64,64) -> (32,128,128) -> (1,256,256)
        Decoder(hidden_channels=[512, 512, 256, 128, 64, 32],
                output_channels=1,
                output_size=sample_size,
                latent_dim=args.latent_dim),
        #kl_weight=args.kl_weight,
        wandb_log=wandb_log, git_hash=args.git_hash, lr=args.lr, seed=seed
    )
    
    print(model.encoder)
    print(model.decoder)
    
    trainer.fit(model, train_loader, val_loader)
    result = trainer.test(model, test_loader)
    
    if args.save_model_dir:
        os.makedirs(args.save_model_dir, exist_ok=True)
    
    if args.save_test_example: # TODO: save test examples to wandb
        model.eval()
        (X, Y) = test_dataset[0]
        X_hat = model.forward(X.unsqueeze(0))[0].squeeze()
        (fig, ax) = dataset.plot_comparison(
            X, X_hat, transform=normalise_x, cbar_unit=r'Pa J$^{-1}$'
        )
        if args.wandb_log:
            wandb.log({'test_example': wandb.Image(fig)})
        if args.save_model_dir:
            fig.savefig(os.path.join(args.save_model_dir, 'test_example1.png'))
    
    wandb.finish()
    
    if args.save_model_dir:
        torch.save(model.state_dict(), os.path.join(args.save_model_dir, 'model.pth'))
        logging.info(f'model saved to {args.save_model_dir}')