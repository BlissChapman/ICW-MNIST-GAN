import MNIST_GAN
import numpy as np
import timeit
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import logger

CUDA = torch.cuda.is_available()
if CUDA:
    print("Using GPU optimizations!")

OUTPUT_DIR = 'output'
TRAINING_STEPS = 200000
BATCH_SIZE = 64
MODEL_DIMENSIONALITY = 64
CRITIC_UPDATES_PER_GENERATOR_UPDATE = 5
LAMBDA = 10
VISUALIZATION_INTERVAL = 1000
NOISE_SAMPLE_LENGTH = 1000
NUM_CLASSES = 10

# ========== Torch Config ==========
torch.manual_seed(1)
if CUDA:
    torch.cuda.manual_seed(1)

# ========== DATA ==========
def noise(size):
    noise = torch.from_numpy(np.random.normal(0.0, size=size)).float()
    if CUDA:
        noise = noise.cuda()
    return noise

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
mnist_dataset = datasets.MNIST('./data',
								train=True,
								download=True,
								transform=transform)
mnist_loader = torch.utils.data.DataLoader(dataset=mnist_dataset,
											batch_size=BATCH_SIZE,
											shuffle=True,
											drop_last=True)

def inf_mnist_train_gen():
    while True:
        for _, (images, labels) in enumerate(mnist_loader):
            # Convert labels to one-hot encoding:
            one_hot_labels = torch.zeros(labels.shape[0], 10)
            for (idx, label) in enumerate(labels):
                one_hot_labels[idx][label] = 1.

            if CUDA:
                images = images.cuda()
                one_hot_labels = one_hot_labels.cuda()

            yield (images, one_hot_labels)

mnist = inf_mnist_train_gen()


# ========== MODELS ==========
generator = MNIST_GAN.Generator(input_size=NOISE_SAMPLE_LENGTH+NUM_CLASSES, output_size=784, dimensionality=MODEL_DIMENSIONALITY, cudaEnabled=CUDA)
critic = MNIST_GAN.Critic(dimensionality=MODEL_DIMENSIONALITY, num_classes=NUM_CLASSES, cudaEnabled=CUDA)

# ========= TRAINING =========
logger = logger.Logger(OUTPUT_DIR)
running_critic_loss = 0.0
running_generator_loss = 0.0
running_batch_start_time = timeit.default_timer()

for training_step in range(1, TRAINING_STEPS+1):
    print("BATCH: [{0}/{1}]\r".format(training_step % VISUALIZATION_INTERVAL, VISUALIZATION_INTERVAL), end='')

    # Critic
    for critic_step in range(CRITIC_UPDATES_PER_GENERATOR_UPDATE):
        images, labels = next(mnist)
        images = Variable(images)

        noise_sample = noise(size=(BATCH_SIZE, NOISE_SAMPLE_LENGTH))
        conditioned_noise_sample = Variable(torch.cat((noise_sample, labels), 1))
        if CUDA:
            conditioned_noise_sample = conditioned_noise_sample.cuda()

        fake_images = generator(conditioned_noise_sample)
        critic_loss = critic.train(images, fake_images, labels, LAMBDA)
        running_critic_loss += critic_loss.data[0]

    # Generator
    noise_sample = noise(size=(BATCH_SIZE, NOISE_SAMPLE_LENGTH))
    conditioned_noise_sample = Variable(torch.cat((noise_sample, labels), 1))
    if CUDA:
        conditioned_noise_sample = conditioned_noise_sample.cuda()

    fake_images = generator(conditioned_noise_sample)
    critic_output = critic(fake_images, labels)
    generator_loss = generator.train(critic_output)
    running_generator_loss += generator_loss.data[0]

    # Visualization
    if training_step % VISUALIZATION_INTERVAL == 0:
        # Timing
        running_batch_elapsed_time = timeit.default_timer() - running_batch_start_time
        running_batch_start_time = timeit.default_timer()
        logger.log_training_step(training_step, TRAINING_STEPS, running_batch_elapsed_time, running_critic_loss, running_generator_loss)

        # Save model weights
        logger.save_model_weights(generator, critic)

        # Visualization
        vis_one_hot_labels = torch.zeros(10, 10)
        if CUDA:
            vis_one_hot_labels = vis_one_hot_labels.cuda()
        for label in range(10):
            vis_one_hot_labels[label][label] = 1
        vis_noise_sample = noise(size=(10, NOISE_SAMPLE_LENGTH))
        vis_conditioned_noise_sample = Variable(torch.cat((vis_noise_sample, vis_one_hot_labels), 1))
        if CUDA:
            vis_conditioned_noise_sample = vis_conditioned_noise_sample.cuda()

        vis_fake_images = generator(vis_conditioned_noise_sample)

        logger.visualize_generated_data(images, vis_fake_images, training_step)
        running_critic_loss = 0.0
        running_generator_loss = 0.0
