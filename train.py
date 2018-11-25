import os
import argparse
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from dataset import get_data
from model import EncoderFC, DecoderFC, DiscriminatorFC
from random import randint
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='Training Model')
parser.add_argument('--cuda', type=str, default='true', help='Set cuda usage')
parser.add_argument('--device_id', type=int, default=0, help='GPU Device ID number')
parser.add_argument('--task_name', type=str, default='flickr8k', help='Set task name')
parser.add_argument('--epoch_size', type=int, default=50, help='Set epoch size')
parser.add_argument('--batch_size', type=int, default=64, help='Set batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Set learning rate for optimizers')
parser.add_argument('--model_path', type=str, default='./models/', help='Set the path for trained models')
parser.add_argument('--latent_size', type=int, default=10, help='Latent size.')
parser.add_argument('--update_interval', type=int, default=3, help='')
parser.add_argument('--log_interval', type=int, default=100, help='Print loss values every log_interval iterations.')
parser.add_argument('--model_save_interval', type=int, default=100, help='Save models every model_save_interval iterations.')

def as_np(data):
    return data.cpu().data.numpy()

def calculate_mean_correlation(domain1_latents, domain2_latents):
    result = 0.0
    nb_features = domain1_latents.shape[1]
    for i in range(nb_features):
        result += pearsonr(domain1_latents[:,i].squeeze(), domain2_latents[:, i].squeeze())[0]
    return result / nb_features

def calculate_auc_score(domain1_latents, domain2_latents):
    y_trues, y_scores = [], []
    nb_positive_pairs = domain1_latents.shape[0]
    for i in range(nb_positive_pairs):
        y_trues.append(1)
        y_scores.append(domain1_latents[i,:].dot(domain2_latents[i,:]))
        # Sample a negative pair
        while True:
            index_1 = randint(0, nb_positive_pairs-1)
            index_2 = randint(0, nb_positive_pairs-1)
            if index_1 != index_2: break
        y_trues.append(0)
        y_scores.append(domain1_latents[index_1,:].dot(domain2_latents[index_2,:]))
    return roc_auc_score(y_trues, y_scores)

def get_gan_loss(discriminator, input_A, input_B, criterion):
    pred_A = discriminator(input_A)
    pred_B = discriminator(input_B)

    labels_dis_A = Variable(torch.zeros([pred_A.size()[0], 1]))
    labels_dis_B = Variable(torch.ones([pred_B.size()[0], 1]))
    labels_gen_A = Variable(torch.ones([pred_A.size()[0], 1]))
    labels_gen_B = Variable(torch.zeros([pred_B.size()[0], 1]))

    if args.cuda == 'true':
        labels_dis_A = labels_dis_A.cuda(args.device_id)
        labels_dis_B = labels_dis_B.cuda(args.device_id)
        labels_gen_A = labels_gen_A.cuda(args.device_id)
        labels_gen_B = labels_gen_B.cuda(args.device_id)

    dis_loss = criterion(pred_A, labels_dis_A) * 0.5 + criterion(pred_B, labels_dis_B) * 0.5
    gen_loss = criterion(pred_A, labels_gen_A) * 0.5 + criterion(pred_B, labels_gen_B) * 0.5

    return dis_loss, gen_loss

def main():
    global args
    args = parser.parse_args()

    cuda = args.cuda
    if cuda == 'true':
        cuda = True
    else:
        cuda = False
    device_id = args.device_id

    task_name = args.task_name
    if task_name == 'flickr8k' or task_name == 'pascal_sentences':
        input_size_A, input_size_B = 2048, 300
    latent_size = args.latent_size

    epoch_size = args.epoch_size
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train_A, train_B, test_A, test_B = get_data(task_name)
    n_batches = int(math.ceil(train_A.size / float(batch_size)))

    encoder_A = EncoderFC(input_size_A, [500, 100],  latent_size)
    decoder_A = DecoderFC(latent_size, [100, 500], input_size_A)
    encoder_B = EncoderFC(input_size_B, [50, 10],  latent_size)
    decoder_B = DecoderFC(latent_size, [10, 50], input_size_B)
    discriminator = DiscriminatorFC(latent_size, [50, 10], 1)
    discriminator_A = DiscriminatorFC(input_size_A, [500, 100], 1)
    discriminator_B = DiscriminatorFC(input_size_B, [500, 100], 1)

    if cuda:
        encoder_A.cuda(device_id)
        decoder_A.cuda(device_id)
        encoder_B.cuda(device_id)
        decoder_B.cuda(device_id)
        discriminator.cuda(device_id)
        discriminator_A.cuda(device_id)
        discriminator_B.cuda(device_id)

    recon_criterion = nn.MSELoss(reduction='sum')
    gan_criterion = nn.BCELoss()

    gen_params = list(encoder_A.parameters()) + list(encoder_B.parameters()) +\
                 list(decoder_A.parameters()) + list(decoder_B.parameters())
    dis_params = list(discriminator.parameters()) + list(discriminator_A.parameters()) + \
                 list(discriminator_B.parameters())

    optim_gen = optim.Adam(gen_params, lr=learning_rate, betas=(0.5,0.999), weight_decay=0.00001)
    optim_dis = optim.Adam(dis_params, lr=learning_rate, betas=(0.5,0.999), weight_decay=0.00001)

    iters = 0

    for epoch in range(epoch_size):
        for i in range(n_batches):
            encoder_A.zero_grad()
            decoder_A.zero_grad()
            encoder_B.zero_grad()
            decoder_B.zero_grad()
            discriminator.zero_grad()
            discriminator_A.zero_grad()
            discriminator_B.zero_grad()

            A = Variable(FloatTensor(train_A.next_items(batch_size)))
            B = Variable(FloatTensor(train_B.next_items(batch_size)))

            if cuda:
                A = A.cuda(device_id)
                B = B.cuda(device_id)

            latent_A = encoder_A(A)
            latent_B = encoder_B(B)
            A_from_latent_A = decoder_A(latent_A)
            A_from_latent_B = decoder_A(latent_B)
            B_from_latent_B = decoder_B(latent_B)
            B_from_latent_A = decoder_B(latent_A)

            # Reconstruction Loss
            recon_loss_A = recon_criterion(A_from_latent_A, A)
            recon_loss_B = recon_criterion(B_from_latent_B, B)

            # Cycle Loss
            ABA = decoder_A(encoder_B(B_from_latent_A))
            BAB = decoder_B(encoder_A(A_from_latent_B))
            cycle_loss_A = recon_criterion(ABA, A)
            cycle_loss_B = recon_criterion(BAB, B)

            # Gan Loss
            dis_loss_latent, gen_loss_latent = get_gan_loss(discriminator, latent_A, latent_B, gan_criterion)
            dis_loss_A, gen_loss_A = get_gan_loss(discriminator_A, A_from_latent_A, A_from_latent_B, gan_criterion)
            dis_loss_B, gen_loss_B = get_gan_loss(discriminator_B, B_from_latent_A, B_from_latent_B, gan_criterion)

            dis_loss_total = dis_loss_latent + dis_loss_A + dis_loss_B
            gen_loss_total = 0.001 * (recon_loss_A + recon_loss_B) + \
                             0.001 * (cycle_loss_A + cycle_loss_B) + \
                             (gen_loss_latent + gen_loss_A + gen_loss_B)
            if iters % args.update_interval == 0:
                dis_loss_total.backward()
                optim_dis.step()
            else:
                gen_loss_total.backward()
                optim_gen.step()

            if iters % args.log_interval == 0:
                print("---------------------")
                print("iters:", iters)
                print("GEN Total Loss:", as_np(gen_loss_total.mean()))
                print("DIS Total Loss:", as_np(dis_loss_total.mean()))
                print("RECON Loss:", as_np(recon_loss_A.mean()), as_np(recon_loss_B.mean()))
                print("CYCLE Loss:", as_np(cycle_loss_A.mean()), as_np(cycle_loss_B.mean()))

                encoder_A.eval()
                encoder_B.eval()
                domainA_data = Variable(FloatTensor(np.asarray(test_A.items)))
                domainB_data = Variable(FloatTensor(np.asarray(test_B.items)))
                if cuda:
                    domainA_data = domainA_data.cuda(device_id)
                    domainB_data = domainB_data.cuda(device_id)
                domainA_latents = as_np(encoder_A(domainA_data))
                domainB_latents = as_np(encoder_B(domainB_data))
                mean_correlation = calculate_mean_correlation(domainA_latents, domainB_latents)
                auc_score = calculate_auc_score(domainA_latents, domainB_latents)
                print("Mean Correlation: {}".format(mean_correlation))
                print("AUC Score: {}".format(auc_score))
                encoder_A.train()
                encoder_B.train()

                sys.stdout.flush()

            if iters > 0 and iters % args.model_save_interval == 0:
                torch.save(encoder_A, os.path.join(model_path, 'model_encoder_A'))
                torch.save(encoder_B, os.path.join(model_path, 'model_encoder_B'))
                torch.save(decoder_A, os.path.join(model_path, 'model_decoder_A'))
                torch.save(decoder_B, os.path.join(model_path, 'model_decoder_B'))
                torch.save(discriminator, os.path.join(model_path, 'model_dis'))
                torch.save(discriminator_A, os.path.join(model_path, 'model_dis_A'))
                torch.save(discriminator_B, os.path.join(model_path, 'model_dis_B'))

            iters += 1

if __name__=="__main__":
    main()
