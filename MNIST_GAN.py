import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad


class Generator(nn.Module):

    def __init__(self, input_size, output_size, dimensionality, cudaEnabled):
        super(Generator, self).__init__()

        # output_size = input_size + (kernel_size - 1)
        self.fc_1 = nn.Linear(input_size, 4*16*16*dimensionality)
        self.deconv_1 = nn.ConvTranspose2d(4*dimensionality, 2*dimensionality, (5, 5))
        self.bn_1 = nn.BatchNorm2d(2*dimensionality)
        self.deconv_2 = nn.ConvTranspose2d(2*dimensionality, dimensionality, (5, 5))
        self.deconv_3 = nn.ConvTranspose2d(dimensionality, 1, (5, 5))

        self.input_size = input_size
        self.output_size = output_size
        self.dimensionality = dimensionality
        self.cudaEnabled = cudaEnabled
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=(0.5, 0.9))

        if cudaEnabled:
            self.cuda()

    def forward(self, x):
        x = x.view(-1, 1, self.input_size)
        out = self.fc_1(x)
        out = F.leaky_relu(out, inplace=True)
        out = out.view(-1, 4*self.dimensionality, 16, 16)
        out = F.dropout(out, 0.3)

        out = self.deconv_1(out)
        out = F.leaky_relu(out, inplace=True)
        out = F.dropout(out, 0.3)
        out = self.bn_1(out)

        out = self.deconv_2(out)
        out = F.leaky_relu(out, inplace=True)
        out = F.dropout(out, 0.3)

        out = self.deconv_3(out)
        out = F.tanh(out)
        out = out.view(-1, self.output_size)

        return out

    def train(self, critic_outputs):
        self.zero_grad()
        g_loss = -torch.mean(critic_outputs)
        g_loss.backward()
        self.optimizer.step()
        return g_loss


class Critic(nn.Module):

    def __init__(self, dimensionality, num_classes, cudaEnabled):
        super(Critic, self).__init__()
        # (W−F+2P)/S+1
        self.conv_1 = nn.Conv2d(1, dimensionality, (2, 2), stride=2)
        self.bn_1 = nn.BatchNorm2d(dimensionality)
        self.conv_2 = nn.Conv2d(dimensionality, 2*dimensionality, (2, 2), stride=2)
        self.bn_2 = nn.BatchNorm2d(2*dimensionality)
        self.conv_3 = nn.Conv2d(2*dimensionality, 4*dimensionality, (3, 3), stride=2)
        self.fc_1 = nn.Linear(3*3*4*dimensionality+num_classes, 1)

        self.dimensionality = dimensionality
        self.cudaEnabled = cudaEnabled
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4, betas=(0.5, 0.9))

        if cudaEnabled:
            self.cuda()

    def forward(self, x, y):
        x = x.view(-1, 1, 28, 28)

        out = self.conv_1(x)
        out = F.leaky_relu(out, inplace=True)
        out = F.dropout(out, 0.3)
        out = self.bn_1(out)

        out = self.conv_2(out)
        out = F.leaky_relu(out, inplace=True)
        out = F.dropout(out, 0.3)
        out = self.bn_2(out)

        out = self.conv_3(out)
        out = F.leaky_relu(out, inplace=True)
        out = F.dropout(out, 0.3)

        out = out.view(out.shape[0], 3*3*4*self.dimensionality)
        labels = Variable(y)
        out = torch.cat((out, labels), 1)
        out = self.fc_1(out)
        out = F.leaky_relu(out, inplace=True)

        return out

    def train(self, real_images, fake_images, labels, LAMBDA):
        # Housekeeping
        self.zero_grad()

        # Compute gradient penalty:
        random_samples = torch.rand(real_images.size())
        interpolated_random_samples = random_samples * real_images.data.cpu() + ((1 - random_samples) * fake_images.data.cpu())
        interpolated_random_samples = Variable(interpolated_random_samples, requires_grad=True)
        if self.cudaEnabled:
            interpolated_random_samples = interpolated_random_samples.cuda()

        critic_random_sample_output = self(interpolated_random_samples, labels)
        grad_outputs = torch.ones(critic_random_sample_output.size())
        if self.cudaEnabled:
            grad_outputs = grad_outputs.cuda()

        gradients = grad(outputs=critic_random_sample_output,
        					inputs=interpolated_random_samples,
        					grad_outputs=grad_outputs,
        					create_graph=True,
        					retain_graph=True,
        					only_inputs=True)[0]
        if self.cudaEnabled:
            gradients = gradients.cuda()

        gradient_penalty = LAMBDA * ((gradients.norm(2) - 1) ** 2).mean()
        if self.cudaEnabled:
            gradient_penalty = gradient_penalty.cuda()

        # Critic output:
        critic_real_output = self(real_images, labels)
        critic_fake_output = self(fake_images, labels)

        # Compute loss
        critic_loss = -(torch.mean(critic_real_output) - torch.mean(critic_fake_output)) + gradient_penalty
        critic_loss.backward()

        # Optimize critic's parameters
        self.optimizer.step()

        return critic_loss
