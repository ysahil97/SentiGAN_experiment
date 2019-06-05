import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Dis_Data_loader
import pickle
import os
from generator import Generator
from discriminator import Discriminator
import matplotlib.pyplot as plt

# from rollout import ROLLOUT


#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 200 # embedding dimension
HIDDEN_DIM = 200 # hidden state dimension of lstm cell
MAX_SEQ_LENGTH = 17  # max sequence length
BATCH_SIZE = 64


#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64


#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 2000
dataset_path = "../../data/movie/"
emb_dict_file = dataset_path + "imdb_word.vocab"

# imdb corpus
imdb_file_txt = dataset_path + "imdb/imdb_sentences.txt"
imdb_file_id = dataset_path + "imdb/imdb_sentences.id"

# sstb corpus
sst_pos_file_txt = dataset_path + 'sstb/sst_pos_sentences.txt'
sst_pos_file_id = dataset_path + 'sstb/sst_pos_sentences.id'
sst_neg_file_txt = dataset_path + 'sstb/sst_neg_sentences.txt'
sst_neg_file_id = dataset_path + 'sstb/sst_neg_sentences.id'


eval_file_1 = 'save/eval_file1.txt'
eval_file_2 = 'save/eval_file2.txt'
eval_text_file = 'save/eval_text_file.txt'
negative_file = 'save/generator_sample.txt'
infer_file = 'save/infer/'


def generate_samples(sess, trainable_model, generated_num, output_file, vocab_list, if_log=False, epoch=0):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num)):
        generated_samples.extend(trainable_model.generate(sess))

    if if_log:
        mode = 'a'
        if epoch == 0:
            mode = 'w'
        with open(eval_text_file, mode) as fout:
            # id_str = 'epoch:%d ' % epoch
            for poem in generated_samples:
                poem = list(poem)
                if 2 in poem:
                    poem = poem[:poem.index(2)]
                buffer = ' '.join([vocab_list[x] for x in poem]) + '\n'
                fout.write(buffer)

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            poem = list(poem)
            if 2 in poem:
                poem = poem[:poem.index(2)]
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def generate_infer(sess, trainable_model, epoch, gen_num, vocab_list):
    generated_samples = []
    for _ in range(int(100)):
        # generated_samples.extend(trainable_model.infer(sess))
        generated_samples.extend(trainable_model.generate(sess))
    file = infer_file+str(epoch)+'_'+str(gen_num)+'.txt'
    with open(file, 'w') as fout:
        for poem in generated_samples:
            poem = list(poem)
            if 2 in poem:
                poem = poem[:poem.index(2)]
            buffer = ' '.join([vocab_list[x] for x in poem]) + '\n'
            fout.write(buffer)
    print("%s saves" % file)
    return


def produce_samples(generated_samples):
    produces_sample = []
    for poem in generated_samples:
        poem_list = []
        for ii in poem:
            if ii == 0:  # _PAD
                continue
            if ii == 2:  # _EOS
                break
            poem_list.append(ii)
        produces_sample.append(poem_list)
    return produces_sample


def load_emb_data(emb_dict_file):
    word_dict = {}
    word_list = []
    item = 0
    with open(emb_dict_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            word = line.strip()
            word_dict[word] = item
            item += 1
            word_list.append(word)
    length = len(word_dict)
    print("Load embedding success! Num: %d" % length)
    return word_dict, length, word_list


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(2):  # data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():

    # load embedding info
    vocab_dict, vocab_size, vocab_list = load_emb_data(emb_dict_file)

    # prepare data
    pre_train_data_loader = []
    pre_train_data_loader_1 = Gen_Data_loader(BATCH_SIZE, vocab_dict)
    pre_train_data_loader_1.create_batches([imdb_file_id, sst_pos_file_id])
    pre_train_data_loader.append(pre_train_data_loader_1)

    # prepare data
    pre_train_data_loader_2 = Gen_Data_loader(BATCH_SIZE, vocab_dict)
    pre_train_data_loader_2.create_batches([imdb_file_id, sst_neg_file_id])
    pre_train_data_loader.append(pre_train_data_loader_2)

    gen_data_loader = []
    gen_data_loader_1 = Gen_Data_loader(BATCH_SIZE, vocab_dict)
    gen_data_loader_1.create_batches([sst_pos_file_id])
    gen_data_loader.append(gen_data_loader_1)

    gen_data_loader_2 = Gen_Data_loader(BATCH_SIZE, vocab_dict)
    gen_data_loader_2.create_batches([sst_neg_file_id])
    gen_data_loader.append(gen_data_loader_2)

    dis_data_loader = Dis_Data_loader(BATCH_SIZE, vocab_dict, MAX_SEQ_LENGTH)

    # build model
    # num_emb, vocab_dict, batch_size, emb_dim, num_units, sequence_length
    generators = []
    generator1 = Generator(vocab_size, vocab_dict, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, MAX_SEQ_LENGTH, 0)
    generators.append(generator1)
    generator2 = Generator(vocab_size, vocab_dict, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, MAX_SEQ_LENGTH, 1)

    generators.append(generator2)
    discriminator = Discriminator(sequence_length=MAX_SEQ_LENGTH, num_classes=2,
                                  vocab_size=vocab_size,
                                  embedding_size=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                  l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    log = open('save/experiment-log.txt', 'w')

    buffer = 'Start pre-training generator...'
    print(buffer)
    log.write(buffer + '\n')
    for epoch in range(150):  #120
        train_loss = [0,0]
        for gen_num in range(len(generators)):
            train_loss[gen_num] = pre_train_epoch(sess, generators[gen_num], pre_train_data_loader[gen_num])
        if epoch % 5 == 0:
            generate_samples(sess, generators[0], 1, eval_file_1, vocab_list, if_log=True, epoch=epoch)
            generate_samples(sess, generators[1], 1, eval_file_2, vocab_list, if_log=True, epoch=epoch)
            print('    pre-train epoch ', epoch, 'train_loss(Gen_1) ', train_loss[0], 'train_loss(Gen_2) ', train_loss[1])
            buffer = '    epoch:\t' + str(epoch) + '\tnll(gen1):\t' + str(train_loss[0]) + '\tnll(gen2):\t' + str(train_loss[1]) + '\n'
            log.write(buffer)

    buffer = 'Start pre-training discriminator...'
    print(buffer)
    log.write(buffer)
    for _ in range(10):   # 10
        generate_samples(sess, generators[0], 70, negative_file, vocab_list)
        generate_samples(sess, generators[1], 70, negative_file, vocab_list)
        dis_data_loader.load_train_data([sst_pos_file_id, sst_neg_file_id], [negative_file])
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: dis_dropout_keep_prob,
                }
                d_loss, d_acc, _ = sess.run([discriminator.loss, discriminator.accuracy, discriminator.train_op], feed)
        buffer = "discriminator loss %f acc %f" % (d_loss, d_acc)
        print(buffer)
        log.write(buffer + '\n')

    print("Start Adversarial Training...")
    log.write('adversarial training...')
    plt.ioff()
    gen_loss = list()
    disc_loss = list()
    for i in range(len(generators)):
        gen_loss.append(list())

    for total_batch in range(TOTAL_BATCH):
        # Train the generators
        rewards_loss = [0,0]
        for gen_num in range(len(generators)):
            for it in range(2):
                # print("1")
                samples = generators[gen_num].generate(sess)
                samples = produce_samples(samples)
                # print("2")
                rewards = generators[gen_num].get_reward(sess, samples, 16, discriminator)
                # print("3")
                a = str(samples[0])
                b = str(rewards[0])
                # rewards = change_rewards(rewards)
                # c = str(rewards[0])
                d = build_from_ids(samples[0], vocab_list)
                buffer = "%s\n%s\n%s\n\n" % (d, a, b)
                print(buffer)
                log.write(buffer)

                # print("4")
                rewards_loss[gen_num] = generators[gen_num].update_with_rewards(sess, samples, rewards)
                gen_loss[gen_num].append(rewards_loss[gen_num])
                # print("5")
                # good rewards
                # good_samples = gen_data_loader.next_batch()
                # rewards = np.array([[0.0001] * SEQ_LENGTH] * BATCH_SIZE)
                # a = str(good_samples[0])
                # b = str(rewards[0])
                # buffer = "%s\n%s\n\n" % (a, b)
                # print(buffer)
                # log.write(buffer)
                # rewards_loss = generator.update_with_rewards(sess, good_samples, rewards, START_TOKEN)

                # little1 good reward
                little1_samples = gen_data_loader[gen_num].next_batch()
                rewards = generators[gen_num].get_reward(sess, little1_samples, 16, discriminator)
                a = str(little1_samples[0])
                b = str(rewards[0])
                buffer = "%s\n%s\n\n" % (a, b)
                # print(buffer)
                log.write(buffer)
                rewards_loss[gen_num] = generators[gen_num].update_with_rewards(sess, little1_samples, rewards)
                gen_loss[gen_num].append(rewards_loss[gen_num])
            # generate_infer(sess, generator, epoch, vocab_list)

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(sess, generators[0], 120, eval_file_1, vocab_list, if_log=True)
            generate_samples(sess, generators[1], 120, eval_file_2, vocab_list, if_log=True)
            generate_infer(sess, generators[0], total_batch, 0, vocab_list)
            generate_infer(sess, generators[1], total_batch, 1, vocab_list)
            # generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            # likelihood_data_loader.create_batches(eval_file)
            # test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            buffer = 'reward-train epoch %s train loss(1) %s train loss(2) %s' % (str(total_batch), str(rewards_loss[0]), str(rewards_loss[1]))
            print(buffer)
            log.write(buffer + '\n')
            generators[0].save_model(sess)
            generators[1].save_model(sess)
            fig = plt.figure()
            plt.plot(gen_loss[0])
            plt.savefig('./save/plots/gen_1.png')
            plt.close(fig)
            fig = plt.figure()
            plt.plot(gen_loss[1])
            plt.savefig('./save/plots/gen_2.png')
            plt.close(fig)



        # Train the discriminator
        begin = True
        for _ in range(1):
            generate_samples(sess, generators[0], 70, negative_file, vocab_list)
            generate_samples(sess, generators[1], 70, negative_file, vocab_list)
            dis_data_loader.load_train_data([sst_pos_file_id, sst_neg_file_id], [negative_file])
            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob,
                    }
                    d_loss, d_acc, _ = sess.run([discriminator.loss, discriminator.accuracy, discriminator.train_op],
                                                feed)
                    disc_loss.append(d_loss)
                    if (total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1) and begin:
                        buffer = "discriminator loss %f acc %f\n" % (d_loss, d_acc)
                        print(buffer)
                        log.write(buffer)
                        begin = False
                        fig = plt.figure()
                        plt.plot(disc_loss)
                        plt.savefig('./save/plots/disc.png')
                        plt.close(fig)

        # pretrain
        for _ in range(10):
            train_loss = pre_train_epoch(sess, generators[0], pre_train_data_loader[0])
            train_loss = pre_train_epoch(sess, generators[1], pre_train_data_loader[1])

# def change_rewards(rewards):
#     ans = []
#     for item in rewards:
#         ans_i = []
#         last_v = 0.0
#         for j in range(len(item)):
#             ans_i.append(max(0.0, item[j] - last_v))
#             last_v = item[j]
#         ans.append(ans_i)
#     return ans


def build_from_ids(vv, vocab_list):
    a = []
    for i in vv:
        a.append(vocab_list[i])
    return(' '.join(a))


if __name__ == '__main__':
    main()


