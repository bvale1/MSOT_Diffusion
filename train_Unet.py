import argparse, wandb, logging, torch, os, json, random
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from pytorch_lightning.loggers import WandbLogger
from custom_pytorch_utils.custom_transforms import *
from custom_pytorch_utils.custom_datasets import *
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from pytorch_utils import *
from Unet_lightning_wrapper import UnetLightning


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='', help='path to the root directory of the dataset')
    parser.add_argument('--git_hash', type=str, default='None', help='optional, git hash of the current commit for reproducibility')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_test_example', help='disable save test examples to wandb', action='store_false')
    parser.add_argument('--seed', type=int, default=None, help='seed for reproducibility')

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
    
    with open(os.path.join(os.path.dirname(args.root_dir), 'config.json'), 'r') as f:
        config = json.load(f) # <- dataset config contains normalisation parameters
    
    normalise_x = MeanStdNormalise(
        torch.Tensor(config['image_normalisation_params']['mean']),
        torch.Tensor(config['image_normalisation_params']['std'])
    )
    x_transform = transforms.Compose([
        ReplaceNaNWithZero(), 
        normalise_x
    ])
    
    normalise_y = MeanStdNormalise(
        torch.Tensor(config['concentration_normalisation_params']['mean']),
        torch.Tensor(config['concentration_normalisation_params']['std'])
    )
    y_transform = transforms.Compose([
        ReplaceNaNWithZero(),
        normalise_y
    ])
    Y_mean = torch.Tensor(config['concentration_normalisation_params']['mean'])
    
    wandb.login()
    wandb_log = WandbLogger(
        project='BphPSEG', name='Unet', save_code=True, reinit=True
    )
    trainer = pl.Trainer.from_argparse_args(
        args, log_every_n_steps=1, check_val_every_n_epoch=1, accelerator='gpu',
        devices=1, max_epochs=args.epochs, deterministic=True, logger=wandb_log
    )
    
    dataset = ReconstructAbsorbtionDataset(
        args.root_dir, gt_type='mu_a', X_transform=x_transform, Y_transform=y_transform
    )
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(42) # train/val/test sets are always the same
    )
    print(f'train: {len(train_dataset)}, val: {len(val_dataset)}, test: \
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
    
    model = UnetLightning(
        smp.Unet(
            encoder_name='resnet101', encoder_weights='imagenet',
            in_channels=1, classes=1,
        ),
        y_transform=normalise_y, y_mean=Y_mean,
        wandb_log=wandb_log, git_hash=args.git_hash, seed=seed
    )
    
    print(model.net)
    
    trainer.fit(model, train_loader, val_loader)
    result = trainer.test(model, test_loader)
    
    if args.save_test_example: # TODO: save test examples to wandb
        model.eval()
        # test example idx. 6 is 'c143423.p31' when 42 is sampler seed
        #(X, Y) = test_dataset[6]
        #Y_hat = model.forward(X.unsqueeze(0)).squeeze()
        #(fig, ax) = dataset.plot_sample(X, Y, Y_hat, y_transform=normalise_y, x_transform=normalise_x)
        #if args.wandb_log:
        #    wandb.log({'test_example': wandb.Image(fig)})