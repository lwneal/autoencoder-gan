import argparse
import numpy as np
import os
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd

import model
from logutil import TimeSeries
import imutil

device = torch.device("cuda")

print('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--latent_size', type=int, default=16)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--lambda_gan', type=float, default=0.1)
parser.add_argument('--huber_scale', type=float, default=1.0)
parser.add_argument('--disc_iters', type=int, default=5)

args = parser.parse_args()


from dataloader import CustomDataloader
loader = CustomDataloader(args.dataset, batch_size=args.batch_size, image_size=64)


print('Building model...')

discriminator = model.Discriminator().to(device)
generator = model.Generator(args.latent_size).to(device)
encoder = model.Encoder(args.latent_size).to(device)

optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))
optim_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))
optim_gen_gan = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))
optim_enc = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.lr, betas=(0.0,0.9))

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
scheduler_g_gan = optim.lr_scheduler.ExponentialLR(optim_gen_gan, gamma=0.99)
scheduler_e = optim.lr_scheduler.ExponentialLR(optim_enc, gamma=0.99)

print('\t...Finished building model')


def sample_z(batch_size, z_dim):
    # Normal Distribution
    z = torch.randn(batch_size, z_dim)
    z = normalize_vector(z)
    return z.to(device)


def normalize_vector(x, eps=.0001):
    norm = torch.norm(x, p=2, dim=1) + eps
    return x / norm.expand(1, -1).t()


def to_np(tensor):
    return tensor.detach().cpu().numpy()


def train(epoch, ts, max_batches=100):
    for i, (current_frame, labels) in enumerate(islice(loader, max_batches)):
        discriminator.train()
        encoder.eval()
        generator.eval()

        optim_disc.zero_grad()
        optim_gen.zero_grad()
        optim_enc.zero_grad()

        # Update discriminator
        z = sample_z(args.batch_size, args.latent_size)
        d_real = 1.0 - discriminator(current_frame)
        d_fake = 1.0 + discriminator(generator(z))
        disc_loss = nn.ReLU()(d_real).mean() + nn.ReLU()(d_fake).mean()
        ts.collect('Disc Loss', disc_loss)
        ts.collect('Disc (Real)', d_real.mean())
        ts.collect('Disc (Fake)', d_fake.mean())
        disc_loss.backward()
        optim_disc.step()

        encoder.train()
        generator.train()

        # Update generator (based on output of discriminator)
        optim_gen.zero_grad()
        z = sample_z(args.batch_size, args.latent_size)
        d_gen = 1.0 - discriminator(generator(z))
        gen_loss = nn.ReLU()(d_gen).mean()
        # Alternative: If you want to only make reconstructions realistic
        #d_gen = 1.0 - discriminator(generator(encoder(current_frame)))
        gen_loss.backward()
        optim_gen.step()

        # For Improved Wasserstein GAN:
        # gp_loss = calc_gradient_penalty(discriminator, ...)
        # gp_loss.backward()

        # Reconstruct pixels
        optim_enc.zero_grad()
        optim_gen.zero_grad()

        encoded = encoder(current_frame)
        reconstructed = generator(encoded)
        # Huber loss
        reconstruction_loss = F.smooth_l1_loss(reconstructed * args.huber_scale, current_frame * args.huber_scale)
        # Alternative: Good old-fashioned MSE loss
        #reconstruction_loss = torch.sum((reconstructed - current_frame)**2)
        ts.collect('Reconst Loss', reconstruction_loss)
        ts.collect('Z variance', encoded.var(0).mean())
        ts.collect('Reconst Pixel variance', reconstructed.var(0).mean())
        ts.collect('Z[0] mean', encoded[:,0].mean().item())

        reconstruction_loss.backward()

        optim_enc.step()
        optim_gen.step()

        if i % args.disc_iters == 0:
            # GAN update for realism
            optim_gen_gan.zero_grad()
            z = sample_z(args.batch_size, args.latent_size)
            generated = generator(z)
            gen_loss = -discriminator(generated).mean() * args.lambda_gan
            ts.collect('Generated pixel variance', generated.var(0).mean())
            ts.collect('Gen Loss', gen_loss)
            gen_loss.backward()
            optim_gen_gan.step()

        ts.print_every(n_sec=4)

    filename = 'reconstruction_epoch_{:04d}.png'.format(epoch)
    imutil.show([current_frame, encoded], caption='Real vs. Reconstruction', filename=filename)

    scheduler_e.step()
    scheduler_d.step()
    scheduler_g.step()
    scheduler_g_gan.step()
    print(ts)


def main():
    batches_per_epoch = 200
    ts_train = TimeSeries('Training', batches_per_epoch * args.epochs)
    for epoch in range(args.epochs):
        print('starting epoch {}'.format(epoch))
        train(epoch, ts_train, batches_per_epoch)
        print(ts_train)


if __name__ == '__main__':
    main()
