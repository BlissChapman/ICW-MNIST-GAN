import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import torch

class Logger:

    def __init__(self, output_dir):
        # Remove and recreate output_dirs
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir)

        self.output_dir = output_dir
        self.model_weights_dir = "{0}/model_weights".format(output_dir)
        self.real_images_dir = "{0}/real_images".format(output_dir)
        self.generated_images_dir = "{0}/generated_images".format(output_dir)
        self.real_distributions_dir = "{0}/real_distributions".format(output_dir)
        self.generated_distributions_dir = "{0}/generated_distributions".format(output_dir)

        os.makedirs(self.model_weights_dir)
        os.makedirs(self.real_images_dir)
        os.makedirs(self.generated_images_dir)
        os.makedirs(self.real_distributions_dir)
        os.makedirs(self.generated_distributions_dir)

        # Set up lists to hold loss histories
        self.critic_losses = []
        self.generator_losses = []

    def save_model_weights(self, generator, critic):
        torch.save(generator.state_dict(), "{0}/generator".format(self.model_weights_dir))
        torch.save(critic.state_dict(), "{0}/critic".format(self.model_weights_dir))

    def log_training_step(self, training_step, num_training_steps, elapsed_training_time, critic_loss, generator_loss):
        # Loss histories
        self.critic_losses.append(critic_loss)
        self.generator_losses.append(generator_loss)
        plt.plot(self.critic_losses)
        plt.plot(self.generator_losses)
        plt.legend(['Critic Loss', 'Generator Loss'])
        plt.savefig('{0}/losses'.format(self.output_dir))
        plt.close()

        # Estimated time remaining
        num_training_steps_remaining = num_training_steps - training_step
        estimated_minutes_remaining = (num_training_steps_remaining*elapsed_training_time)/60.0

        print("===== TRAINING STEP {} | ~{:.0f} MINUTES REMAINING =====".format(training_step, estimated_minutes_remaining))
        print("CRITIC LOSS:     {0}".format(critic_loss))
        print("GENERATOR LOSS:  {0}\n".format(generator_loss))

    def visualize_generated_data(self, real_images, fake_images, training_step):
        # Images:
        generator_sample_images = fake_images.data.cpu().numpy().reshape(-1, 28, 28)
        Logger.visualize_ten_images(generator_sample_images, '{0}/step_{1}'.format(self.generated_images_dir, training_step))

        real_sample_images = real_images.data.cpu().numpy().reshape(-1, 28, 28)
        Logger.visualize_ten_images(real_sample_images, '{0}/step_{1}'.format(self.real_images_dir, training_step))

        # Distributions
        generator_sample_distributions = fake_images.data.cpu().numpy().reshape(-1, 784)
        Logger.visualize_ten_distributions(generator_sample_distributions, '{0}/step_{1}'.format(self.generated_distributions_dir, training_step))

        real_sample_distributions = real_images.data.cpu().numpy().reshape(-1, 784)
        Logger.visualize_ten_distributions(real_sample_distributions, '{0}/step_{1}'.format(self.real_distributions_dir, training_step))

    def visualize_ten_images(images, output_path):
        plt.figure()
        for i in range(10):
            ax = plt.subplot(3, 4, i+1)
            im = ax.imshow(images[i], cmap="gray")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def visualize_ten_distributions(distributions, output_path):
        plt.figure()
        for i in range(10):
            ax = plt.subplot(3, 4, i+1)
            ax.set_ylim([-1.1, 1.1])
            ax.plot(distributions[i])
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
