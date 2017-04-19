"""
Architecture:

 --top---
 |       |
pen     labels
 |
hid
 |
vis
"""

from sets import Set
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tf_utils import *
from tensorflow.examples.tutorials.mnist import input_data

numvis = 784 # 28x28 greyscale pixels
numhid = 784
numpen = 784
numtop = 784
numlab = 10
batch_count = 10000
learning_rate = 0.01
log = "log"
CD = 10

###############################################################################################train vishid rbm
# VIS<->HID

sess = tf.Session()

with tf.name_scope('neurons'):
    hid = tf.placeholder('float', [numhid, 1], name='hid')
    vis = tf.placeholder('float', [numvis, 1], name='vis')

# xxxyyy_w = weights from layer xxx to layer yyy
with tf.name_scope('symmetric_weights'):
    vishid_w = tf.Variable(tf.truncated_normal((numhid, numvis), mean=0.0, stddev=0.001), name='vishid_w')
    tf.add_to_collection('weights', vishid_w)

with tf.name_scope('biases'):
    hidvis_b = tf.Variable(tf.truncated_normal((numvis, 1), mean=0.0, stddev=0.001), name='hidvis_b')
    vishid_b = tf.Variable(tf.truncated_normal((numhid, 1), mean=0.0, stddev=0.001), name='vishid_b')
    tf.add_to_collection('biases', hidvis_b)
    tf.add_to_collection('biases', vishid_b)

pixels = tf.placeholder('float', [1, 784], name='pixels')
binary_pixels = tf.ceil(pixels - 0.5)

vis = tf.reshape(binary_pixels, [784, 1])
tmp_vis = vis

vishid_summaries = []

hidprobs = None
hidprobs0 = None

# contrastive divergence
for step in range(CD):
    hidprobs = tf.sigmoid(tf.matmul(vishid_w, tmp_vis) + vishid_b)
    if (0 == step):
        hid = bernoulli_sample(hidprobs)
        hidprobs0 = hidprobs
    else:
        hid = hidprobs

    visprobs = tf.sigmoid(tf.matmul(tf.transpose(vishid_w), hid) + hidvis_b)
    tmp_vis = visprobs
    #tmp_vis = bernoulli_sample(visprobs)


vishid_pos_stats = tf.matmul(hidprobs0, tf.transpose(vis))
vishid_neg_stats = tf.matmul(hidprobs, tf.transpose(tmp_vis))
vishid_w_target = tf.assign_add(vishid_w, learning_rate*(vishid_pos_stats-vishid_neg_stats))
hidvis_b_target = tf.assign_add(hidvis_b, learning_rate*(vis-tmp_vis))
vishid_b_target = tf.assign_add(vishid_b, learning_rate*(hidprobs0-hidprobs))

#cross_entropy = tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits(labels=vis, logits=tf.log(visprobs)))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(vis * tf.log(visprobs)))
vishid_summaries.append(tf.summary.scalar("vishid_data_cross entropy", cross_entropy))

side = 28

#input_images = tf.reshape(vis, [-1, side, side, 1])
#image = tf.slice(input_images, [0, 0, 0, 0], [1, side, side, 1])
#vishid_summaries.append(tf.summary.image("vishid_inputs", image, max_outputs=10))

vishid_w_square = tf.reshape(vishid_w_target, [-1, side, side, 1])
first2vishid_weights = tf.slice(vishid_w_square, [0, 0, 0, 0], [2, side, side, 1])
vishid_summaries.append(tf.summary.image("first2vishid_weights", first2vishid_weights, max_outputs=10))
vishid_summaries.append(tf.summary.histogram("vishid_w_target", vishid_w_target))
vishid_summaries.append(tf.summary.histogram("vishid_b_target", vishid_b_target))
vishid_summaries.append(tf.summary.histogram("hidvis_b_target", hidvis_b_target))

#vishid_b_image = tf.reshape(vishid_b, [-1, side, side, 1])
#vishid_summaries.append(tf.summary.image("vishid_b_image", vishid_b_image, max_outputs=10))
#
#vishid_input_image = tf.reshape(tf.transpose(vis), [-1, side, side, 1])
#vishid_summaries.append(tf.summary.image("vishid_last_input_image", vishid_input_image, max_outputs=10))
#
#hidprobs_image = tf.reshape(tf.transpose(hidprobs), [-1, side, side, 1])
#vishid_summaries.append(tf.summary.image("hidprobs_image", hidprobs_image, max_outputs=10))

#vishid_saver = tf.train.Saver({vishid_w.name: vishid_w, vishid_b.name: vishid_b, hidvis_b.name: hidvis_b})

vishid_saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_writer = tf.summary.FileWriter(log + "/vishid")
merged = tf.summary.merge(vishid_summaries)
print("training vishid rbm...")
for batch in tqdm(range(batch_count)):
    data, labels = mnist.train.next_batch(1)
    updates = [merged, vishid_w_target, vishid_b_target, hidvis_b_target]
    summary, _, _, _ = sess.run(updates, feed_dict={pixels: data})
    train_writer.add_summary(summary, batch)

vishid_saver.save(sess, 'vishid_new_model')

train_writer.close()
sess.close()
##################################################################################################train hidpen rbm
# HID<->PEN


with tf.name_scope('biases'):
    hidpen_b = tf.Variable(tf.truncated_normal((numpen, 1), mean=0.0, stddev=0.001), name='hidpen_b')
    penhid_b = tf.Variable(tf.truncated_normal((numpen, 1), mean=0.0, stddev=0.001), name='penhid_b')
    tf.add_to_collection('biases', hidpen_b)
    tf.add_to_collection('biases', penhid_b)

with tf.name_scope('neurons'):
    pen = tf.placeholder('float', [numpen, 1], name='pen')

#with tf.name_scope('weights'):
    #vishid_w = tf.Variable(-1, validate_shape=False, name='vishid_w')

#### Restore variables from previous model
sess = tf.Session()
vishid_restorer = tf.train.import_meta_graph('vishid_new_model.meta')
vishid_restorer.restore(sess, tf.train.latest_checkpoint('./'))

#hidvis_b_val = tf.reshape([v for v in tf.global_variables() if v.name == "biases/hidvis_b:0"], [numvis, 1])
#vishid_w_val = get_tensor_value('vishid_w', 'weights')
#hidvis_b = tf.Variable(hidvis_b_val, name="hidvis_b")
#vishid_b_val = tf.reshape([v for v in tf.global_variables() if v.name == "biases/vishid_b:0"], [numhid, 1])
vishid_b_val = get_tensor_value(vishid_b.name, 'biases')
vishid_b_init = tf.Variable(vishid_b_val, name="vishid_b_init")
tf.add_to_collection('biases', vishid_b_init)

vishid_w_val = get_tensor_value(vishid_w.name, 'weights')
#vishid_w_val = tf.reshape([v for v in tf.global_variables() if v.name == "symmetric_weights/vishid_w:0"], [numhid, numvis])
vishid_w_init = tf.Variable(vishid_w_val, name="vishid_w_init")
tf.add_to_collection('weights', vishid_w_init)

with tf.name_scope('symmetric_weights'):
    hidpen_w = tf.Variable(vishid_w_val, name='hidpen_w')
    tf.add_to_collection('weights', hidpen_w)

# Run input through lower layer(s):
hidpen_pixels = tf.placeholder('float', [1, 784], name='hidpen_pixels')
hidpen_binary_pixels = tf.ceil(hidpen_pixels - 0.5)
hidpen_vis = tf.reshape(hidpen_binary_pixels, [784, 1])

inputhidprobs = tf.sigmoid(tf.matmul(vishid_w_init, hidpen_vis) + vishid_b_init)
inputhid = bernoulli_sample(inputhidprobs)
#####

tmp_hid = inputhid

hidpen_summaries = []

penhidprobs = None
penprobs = None
penprobs0 = None

# contrastive divergence
for step in range(CD):
    penprobs = tf.sigmoid(tf.matmul(hidpen_w, tmp_hid) + hidpen_b)
    if (0 == step):
        pen = bernoulli_sample(penprobs)
        penprobs0 = penprobs
    else:
        pen = penprobs

    penhidprobs = tf.sigmoid(tf.matmul(tf.transpose(hidpen_w), pen) + penhid_b)
    tmp_hid = penhidprobs 
    #tmp_hid = bernoulli_sample(penhidprobs)

hidpen_pos_stats = tf.matmul(penprobs0, tf.transpose(inputhid))
hidpen_neg_stats = tf.matmul(penprobs, tf.transpose(tmp_hid))
hidpen_w_target = tf.assign_add(hidpen_w, learning_rate*(hidpen_pos_stats-hidpen_neg_stats))
penhid_b_target = tf.assign_add(penhid_b, learning_rate*(inputhid-tmp_hid))
hidpen_b_target = tf.assign_add(hidpen_b, learning_rate*(penprobs0-penprobs))

side = 28

#input_probs = tf.reshape(penhidprobs, [-1, side, side, 1])
#first2probs = tf.slice(input_probs, [0, 0, 0, 0], [1, side, side, 1])
#hidpen_summaries.append(tf.summary.image("first2hidpen_probs", first2probs, max_outputs=10))


### previous model inputs
#hidpen_input_from_vishid_biases = tf.reshape(vishid_b_init, [-1, side, side, 1])
#hidpen_summaries.append(tf.summary.image("hidpen_input_from_vishid_biases", hidpen_input_from_vishid_biases, max_outputs=10))
#
#hidpen_input_from_vishid_weights = tf.reshape(vishid_w_init, [-1, side, side, 1])
#first2hidpen_input_from_weights = tf.slice(hidpen_input_from_vishid_weights, [0, 0, 0, 0], [2, side, side, 1])
#hidpen_summaries.append(tf.summary.image("first2hidpen_input_weights", first2hidpen_input_from_weights, max_outputs=10))
#
#hidpen_input_image_from_vishid = tf.reshape(inputhid, [-1, side, side, 1])
#hidpen_summaries.append(tf.summary.image("hidpen_input_image_from_vishid", hidpen_input_image_from_vishid, max_outputs=10))
###

hidpen_w_square = tf.reshape(hidpen_w_target, [-1, side, side, 1])
first2hidpen_weights = tf.slice(hidpen_w_square, [0, 0, 0, 0], [2, side, side, 1])
hidpen_summaries.append(tf.summary.image("first2hidpen_weights", first2hidpen_weights, max_outputs=10))

#hidpen_summaries.append(tf.summary.histogram("hidpen_w_target", hidpen_w_target))
#hidpen_summaries.append(tf.summary.histogram("hidpen_b_target", hidpen_b_target))
#hidpen_summaries.append(tf.summary.histogram("penhid_b_target", penhid_b_target))

hidpen_saver = tf.train.Saver({hidpen_w.name: hidpen_w, hidpen_b.name: hidpen_b, penhid_b.name: penhid_b, vishid_b_init.name: vishid_b_init, hidvis_b.name: hidvis_b, vishid_w_init.name: vishid_w_init})

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
tf.global_variables_initializer().run(session=sess)
train_writer = tf.summary.FileWriter(log + "/hidpen")
merged = tf.summary.merge(hidpen_summaries)
print("training hidpen rbm...")
for batch in tqdm(range(batch_count)):
    data, labels = mnist.train.next_batch(1)
    updates = [merged, hidpen_w_target, hidpen_b_target, penhid_b_target]
    summary, _, _, _ = sess.run(updates, feed_dict={hidpen_pixels: data})
    #summary = sess.run(merged, feed_dict={pixels: data})
    train_writer.add_summary(summary, batch)

hidpen_saver.save(sess, 'hidpen_model')
train_writer.close()
sess.close()

#tf.reset_default_graph()
##########################################################################################train pentop rbm
#  LAB <--|
#         |---->TOP
#  PEN <--|

pentop_var_list = []
with tf.name_scope('symmetric_weights'):
    labtop_w = tf.Variable(tf.truncated_normal((numtop, numlab), mean=0.0, stddev=0.001), name='labtop')
    tf.add_to_collection('weights', labtop_w)
    pentop_var_list.append(labtop_w)

with tf.name_scope('biases'):
    pentop_b = tf.Variable(tf.truncated_normal((numtop, 1), mean=0.0, stddev=0.001), name='pentop_b')
    toppen_b = tf.Variable(tf.truncated_normal((numpen, 1), mean=0.0, stddev=0.001), name='toppen_b')
    toplab_b = tf.Variable(tf.truncated_normal((numlab, 1), mean=0.0, stddev=0.001), name='toplab_b')
    tf.add_to_collection('biases', pentop_b)
    tf.add_to_collection('biases', toppen_b)
    tf.add_to_collection('biases', toplab_b)
    pentop_var_list.append(pentop_b)
    pentop_var_list.append(toppen_b)
    pentop_var_list.append(toplab_b)


with tf.name_scope('neurons'):
    top = tf.placeholder('float', [numtop, 1], name='top')
    lab = tf.placeholder('float', [numlab, 1], name='lab')
    #lab_on_deck = tf.placeholder('float', [1, numlab], name='lab_on_deck')
    #lab_on_deck = tf.constant([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], 'float')
    lab_on_deck = tf.placeholder_with_default(tf.truncated_normal((1, numlab), mean=0.0, stddev=0.001), [1, numlab], name='lab_on_deck')

sess = tf.Session()

######## Restore variables from previous models

# No idea why I can't restore weights multiple times...
#vishid_restorer = tf.train.import_meta_graph('vishid_model.meta')
#vishid_restorer.restore(sess, tf.train.latest_checkpoint('./'))

hidpen_restorer = tf.train.import_meta_graph('hidpen_model.meta')
hidpen_restorer.restore(sess, tf.train.latest_checkpoint('./'))

pentop_vishid_b_init = None
unused_biases = []
variables = Set()
variables.add(vishid_b.name)
biases = tf.get_collection('biases')
for b in biases:
    if (b.name == vishid_b_init.name):
        try:
            variables.remove(vishid_b.name)
            print("Restored " + b.name)
            pentop_vishid_b_init = tf.Variable(b, name="pentop_vishid_b_init")
            pentop_var_list.append(pentop_vishid_b_init)
        except KeyError:
            pass
    else:
        unused_biases.append(b.name)
        #print("No luck, " + b.name + " != " + vishid_b.name)
print("Leftover biases to find: " + str(variables))
print("Unused items in collection: " + str(unused_biases))

hidpen_w_init = None
pentop_vishid_w_init = None
unused_weights = []
variables = Set()
variables.add(vishid_w.name)
variables.add(hidpen_w.name)
weights = tf.get_collection('weights')
print("weights collection:")
for w in weights:
    if (w.name == vishid_w_init.name):
        try:
            variables.remove(vishid_w.name)
            print("Restored " + w.name)
            pentop_vishid_w_init = tf.Variable(w, name="pentop_vishid_w_init")
            pentop_var_list.append(pentop_vishid_w_init)
        except KeyError:
            pass 
    elif (w.name == hidpen_w.name):
        try:
            variables.remove(hidpen_w.name)
            print("Restored " + w.name)
            hidpen_w_init = tf.Variable(w, name="hidpen_w_init")
            pentop_var_list.append(hidpen_w_init)
            with tf.name_scope('symmetric_weights'):
                pentop_w = tf.Variable(hidpen_w, name='pentop_w')
                tf.add_to_collection('weights', pentop_w)
                pentop_var_list.append(pentop_w)
        except KeyError:
            pass 
    else:
        unused_weights.append(w.name)
print("Leftover weights to find: " + str(variables))
print("Unused items in collection: " + str(unused_weights))

# No idea why get_tensor_value() works/doesn't work 50% of the time
hidpen_b_init = tf.Variable(get_tensor_value(hidpen_b.name, 'biases'), name="hidpen_b_init")
pentop_var_list.append(hidpen_b_init)

######## Done restoring previous model variables

# Construct previous model input
pentop_pixels = tf.placeholder('float', [1, 784], name='pentop_pixels')
pentop_binary_pixels = tf.ceil(pentop_pixels - 0.5)
pentop_vis = tf.reshape(pentop_binary_pixels, [784, 1])
pentop_inputhidprobs = tf.sigmoid(tf.matmul(pentop_vishid_w_init, pentop_vis) + pentop_vishid_b_init)
pentop_inputhid = bernoulli_sample(pentop_inputhidprobs)
#
pentop_inputpenprobs = tf.sigmoid(tf.matmul(hidpen_w_init, pentop_vis) + hidpen_b_init)
#pentop_inputpenprobs = tf.sigmoid(tf.matmul(hidpen_w, pentop_vis) + hidpen_b_init)
pentop_inputpen = bernoulli_sample(pentop_inputpenprobs)

# Copy weights from hidpen rbm to top level rbm (pen<->top)

lab = tf.transpose(lab_on_deck)

tmp_pen = pentop_inputpen
tmp_lab = lab

pentop_summaries = []
labtop_summaries = []

topprobs = None
topprobs0 = None

# contrastive divergence
CD = 3
for step in range(CD):
    topprobs = tf.sigmoid((tf.matmul(pentop_w, tmp_pen) + tf.matmul(labtop_w, tmp_lab)) + pentop_b)
    if (0 == step):
        top = bernoulli_sample(topprobs)
        topprobs0 = topprobs
    else:
        top = topprobs

    penprobs = tf.sigmoid(tf.matmul(tf.transpose(pentop_w), top) + toppen_b)
    tmp_pen = penprobs
    #tmp_pen = bernoulli_sample(penprobs)

    labprobs = tf.sigmoid(tf.matmul(tf.transpose(labtop_w), top) + toplab_b)
    tmp_lab = labprobs
    #tmp_lab = bernoulli_sample(labprobs)


pentop_pos_stats = tf.matmul(topprobs0, tf.transpose(pentop_inputpen))
pentop_neg_stats = tf.matmul(topprobs, tf.transpose(tmp_pen))
pentop_w_target = tf.assign_add(pentop_w, learning_rate*(pentop_pos_stats-pentop_neg_stats))
toppen_b_target = tf.assign_add(toppen_b, learning_rate*(pentop_inputpen-tmp_pen))

labtop_pos_stats = tf.matmul(topprobs0, tf.transpose(lab))
labtop_neg_stats = tf.matmul(topprobs, tf.transpose(tmp_lab))
labtop_w_target = tf.assign_add(labtop_w, learning_rate*(labtop_pos_stats-labtop_neg_stats))
toplab_b_target = tf.assign_add(toplab_b, learning_rate*(lab-tmp_lab))

pentop_b_target = tf.assign_add(pentop_b, learning_rate*(topprobs0-topprobs))

side = 28

#input_probs = tf.reshape(penprobs, [-1, side, side, 1])
#first2probs = tf.slice(input_probs, [0, 0, 0, 0], [1, side, side, 1])
#pentop_summaries.append(tf.summary.image("first2pentop_probs", first2probs, max_outputs=10))

#input_weights = tf.reshape(new_weights, [-1, side, side, 1])
#first2weights = tf.slice(input_weights, [0, 0, 0, 0], [1, side, side, 1])
#pentop_summaries.append(tf.summary.image("first2pentop_weights", first2weights, max_outputs=10))


### previous model inputs
#hidpen_input_from_vishid_biases = tf.reshape(pentop_vishid_b_init, [-1, side, side, 1])
#hidpen_summaries.append(tf.summary.image("hidpen_input_from_vishid_biases", hidpen_input_from_vishid_biases, max_outputs=10))
#
#hidpen_input_from_vishid_weights = tf.reshape(pentop_vishid_w_init, [-1, side, side, 1])
#first2hidpen_input_from_weights = tf.slice(hidpen_input_from_vishid_weights, [0, 0, 0, 0], [2, side, side, 1])
#hidpen_summaries.append(tf.summary.image("first2hidpen_input_weights", first2hidpen_input_from_weights, max_outputs=10))
#
#hidpen_input_image_from_vishid = tf.reshape(inputhid, [-1, side, side, 1])
#hidpen_summaries.append(tf.summary.image("hidpen_input_image_from_vishid", hidpen_input_image_from_vishid, max_outputs=10))
###

#input_images = tf.reshape(pentop_inputpen, [-1, side, side, 1])
#first2images = tf.slice(input_images, [0, 0, 0, 0], [1, side, side, 1])
#pentop_summaries.append(tf.summary.image("first2pentop_inputs", first2images, max_outputs=10))
#
pentop_w_square = tf.reshape(pentop_w_target, [-1, side, side, 1])
first2pentop_weights = tf.slice(pentop_w_square, [0, 0, 0, 0], [2, side, side, 1])
pentop_summaries.append(tf.summary.image("first2pentop_weights", first2pentop_weights, max_outputs=10))
#
#pentop_summaries.append(tf.summary.histogram("pentop_w_target", pentop_w_target))
#pentop_summaries.append(tf.summary.histogram("pentop_b_target", pentop_b_target))
#
labtop_summaries.append(tf.summary.histogram("labtop_w_target", labtop_w_target))
#labtop_summaries.append(tf.summary.histogram("toplab_b_target", toplab_b_target))
#
#pentop_summaries.append(tf.summary.histogram("toppen_b_target", toppen_b_target))

pentop_saver = tf.train.Saver({pentop_w.name: pentop_w, labtop_w.name: labtop_w, pentop_b.name: pentop_b, toppen_b.name: toppen_b, toplab_b.name: toplab_b})
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#tf.global_variables_initializer().run(session=sess)
tf.variables_initializer(pentop_var_list).run(session=sess)
train_writer = tf.summary.FileWriter(log + "/pentop")
pentop_merged = tf.summary.merge(pentop_summaries)
labtop_merged = tf.summary.merge(labtop_summaries)
print("training pentop rbm...")
for batch in tqdm(range(batch_count)):
    data, labels = mnist.train.next_batch(1)
    updates = [lab_on_deck, labprobs, pentop_merged, labtop_merged, pentop_w_target, labtop_w_target, pentop_b_target, toppen_b_target, toplab_b_target]
    lod, in_labels, pentop_summary, labtop_summary, _, _, _, _, _ = sess.run(updates, feed_dict={pentop_pixels: data, lab_on_deck: labels})
    #print("predicted lab for pentop training " + str(in_labels))# + "\nlab = " + str())
    #print("lab on deck = " + str(lod))
    train_writer.add_summary(pentop_summary, batch)
    train_writer.add_summary(labtop_summary, batch)

pentop_saver.save(sess, 'pentop_model')
train_writer.close()
#sess.close()

##################################################################################################up down algorithm
#    --top---        ^ Up   |
#    |       |       |      |
#   pen     labels   |      |
#    |               |      |
#   hid              |      |
#    |               |      |
#   vis              |      v Down

updown_var_list = []

#Untie symmetryic weights
with tf.name_scope('recognition_weights'):
    rec_vishid_w = tf.Variable(pentop_vishid_w_init, name='rec_vishid_w')
    rec_hidpen_w = tf.Variable(hidpen_w_init, name='rec_hidpen_w')
    updown_var_list.append(rec_hidpen_w)
    updown_var_list.append(rec_vishid_w)

with tf.name_scope('generative_weights'):
    gen_vishid_w = tf.Variable(pentop_vishid_w_init, name='gen_vishid_w')
    gen_hidpen_w = tf.Variable(hidpen_w_init, name='gen_hidpen_w')
    updown_var_list.append(gen_hidpen_w)
    updown_var_list.append(gen_vishid_w)

with tf.name_scope('recognition_biases'):
    rec_vishid_b = tf.Variable(vishid_b_init, name='rec_vishid_b')
    rec_hidpen_b = tf.Variable(hidpen_b, name='rec_hidpen_b')
    updown_var_list.append(rec_hidpen_b)
    updown_var_list.append(rec_vishid_b)
   
with tf.name_scope('generative_biases'):
    gen_vishid_b = tf.Variable(hidvis_b, name='gen_vishid_b')
    gen_hidpen_b = tf.Variable(hidpen_b, name='gen_hidpen_b')
    updown_var_list.append(gen_hidpen_b)
    updown_var_list.append(gen_vishid_b)

targets = lab

updownsummaries = []

#untie weights
#rec_vishid_w = tf.Variable(pentop_vishid_w_init, name="rec_vishid_w")
#rec_hidpen_w = tf.Variable( name="rec_hidpen_w")
#gen_vishid_w = tf.Variable( name="gen_vishid_w")
#gen_hidpen_w = tf.Variable( name="gen_hidpen_w")
#updown_var_list.append(rec_hidpen_w)
#updown_var_list.append(gen_hidpen_w)
#updown_var_list.append(rec_vishid_w)
#updown_var_list.append(gen_vishid_w)
#
#tf.assign(rec_hidpen_w, hidpen_w_target) 
#tf.assign(gen_hidpen_w, hidpen_w_target)
#tf.assign(rec_vishid_w, vishid_w_target)
#tf.assign(gen_vishid_w, vishid_w_target)

# Perform up pass (wake)
wakehidprobs = tf.sigmoid(tf.matmul(rec_vishid_w, pentop_vis) + rec_vishid_b)
wakehid = bernoulli_sample(wakehidprobs)

wakepenprobs = tf.sigmoid(tf.matmul(rec_hidpen_w, wakehid) + rec_hidpen_b)
wakepen = bernoulli_sample(wakepenprobs)

postopprobs = tf.sigmoid(tf.matmul(pentop_w, wakepen) + tf.matmul(labtop_w, targets) + pentop_b)
postopstates = bernoulli_sample(postopprobs)

poslabtopstats = tf.matmul(postopstates, tf.transpose(targets))
pospentopstats = tf.matmul(postopstates, tf.transpose(wakepen))

# Perform contrastive divergence on top level rbm for CD iterations
negtopstates = postopstates
neglabprobs = None
for step in range(CD):
    negpenprobs = tf.sigmoid(tf.matmul(tf.transpose(pentop_w), negtopstates) + toppen_b)
    negpenstates = bernoulli_sample(negpenprobs)

    neglabprobs = tf.nn.softmax(tf.matmul(tf.transpose(labtop_w), negtopstates) + toplab_b, 0)
    
    negtopprobs = tf.sigmoid(tf.matmul(pentop_w, negpenstates) + tf.matmul(labtop_w, neglabprobs) + pentop_b)
    negtopstates = bernoulli_sample(negtopprobs)

negpentopstats = tf.matmul(negtopstates, tf.transpose(negpenstates))
neglabtopstats = tf.matmul(negtopstates, tf.transpose(neglabprobs))

# Perform down pass (sleep)
sleeppenstates = negpenstates
sleephidprobs = tf.sigmoid(tf.matmul(tf.transpose(gen_hidpen_w), negpenstates) + gen_hidpen_b)
sleephidstates = bernoulli_sample(sleephidprobs)
sleepvisprobs = tf.sigmoid(tf.matmul(tf.transpose(gen_vishid_w), sleephidstates) + gen_vishid_b)

# Predictions
psleeppenstates = tf.sigmoid(tf.matmul(rec_hidpen_w, sleephidstates) + rec_hidpen_b)
psleephidstates = tf.sigmoid(tf.matmul(rec_vishid_w, sleepvisprobs) + rec_vishid_b)
pvisprobs = tf.sigmoid(tf.matmul(tf.transpose(gen_vishid_w), wakehid) + gen_vishid_b)
phidprobs = tf.sigmoid(tf.matmul(tf.transpose(gen_hidpen_w), wakepen) + gen_hidpen_b)

# Update generative weights and biases
gen_vishid_w_target = tf.assign_add(gen_vishid_w, learning_rate*tf.matmul(wakehid, tf.transpose(pentop_vis-pvisprobs)))
gen_vishid_b_target = tf.assign_add(gen_vishid_b, learning_rate * (pentop_vis-pvisprobs))
gen_hidpen_w_target = tf.assign_add(gen_hidpen_w, learning_rate*tf.matmul(wakehid, tf.transpose(wakehid-phidprobs)))
gen_hidpen_b_target = tf.assign_add(gen_hidpen_b, learning_rate * (wakehid-phidprobs))

# Updates to top level rbm (associative memory)
pentop_w_target = tf.assign_add(pentop_w, learning_rate * (pospentopstats-negpentopstats))
labtop_w_target = tf.assign_add(labtop_w, learning_rate * (poslabtopstats-neglabtopstats))
toppen_b_target = tf.assign_add(toppen_b, learning_rate * (wakepen-negpenstates))
toplab_b_target = tf.assign_add(toplab_b, learning_rate * (targets-neglabprobs))
top_b_target = tf.assign_add(pentop_b, learning_rate * (postopstates-negtopstates))

# Updates to recognition weights and biases
rec_hidpen_w_target = tf.assign_add(rec_hidpen_w, learning_rate * tf.matmul(sleephidstates, tf.transpose(sleeppenstates-psleeppenstates)))
rec_hidpen_b_target = tf.assign_add(rec_hidpen_b, learning_rate * (sleeppenstates-psleeppenstates))
rec_vishid_w_target = tf.assign_add(rec_vishid_w, learning_rate * tf.matmul(sleepvisprobs, tf.transpose(sleephidstates-psleephidstates)))
rec_vishid_b_target = tf.assign_add(rec_vishid_b, learning_rate * (sleephidstates-psleephidstates))

data_cross_entropy = tf.reduce_mean(-tf.reduce_sum(pentop_vis * tf.log(sleepvisprobs)))
updownsummaries.append(tf.summary.scalar("data_cross_entropy", data_cross_entropy))

label_cross_entropy = tf.reduce_mean(-tf.reduce_sum(lab * tf.log(neglabprobs)))
updownsummaries.append(tf.summary.scalar("label_cross_entropy", label_cross_entropy))

#neglabprobs_w_square = tf.reshape(neglabprobs, [-1, side, side, 1])
#first2neglabprobs_weights = tf.slice(neglabprobs_w_square, [0, 0, 0, 0], [2, side, side, 1])
#updownsummaries.append(tf.summary.image("first2neglabprobs_weights", first2neglabprobs_weights, max_outputs=10))

#wakehidprobs_square = tf.reshape(wakehidprobs, [-1, side, side, 1])
#first2wakehidprobseights = tf.slice(wakehidprobs_square, [0, 0, 0, 0], [2, side, side, 1])
#updownsummaries.append(tf.summary.image("first2wakehidprobs", first2wakehidprobseights, max_outputs=10))

rec_vishid_w_square = tf.reshape(rec_vishid_w_target, [-1, side, side, 1])
first2rec_vishid_weights = tf.slice(rec_vishid_w_square, [0, 0, 0, 0], [2, side, side, 1])
updownsummaries.append(tf.summary.image("first2rec_vishid_weights", first2rec_vishid_weights, max_outputs=10))

gen_vishid_w_square = tf.reshape(gen_vishid_w_target, [-1, side, side, 1])
first2gen_vishid_weights = tf.slice(gen_vishid_w_square, [0, 0, 0, 0], [2, side, side, 1])
updownsummaries.append(tf.summary.image("first2gen_vishid_weights", first2gen_vishid_weights, max_outputs=10))

#rec_vishid_w_square = tf.reshape(rec_vishid_w_target, [-1, side, side, 1])
#first2rec_vishid_weights = tf.slice(rec_vishid_w_square, [0, 0, 0, 0], [2, side, side, 1])
#updownsummaries.append(tf.summary.image("first2rec_vishid_weights", first2rec_vishid_weights, max_outputs=10))

updownsummaries.append(tf.summary.histogram("neglabprobs", neglabprobs))
updownsummaries.append(tf.summary.histogram("rec_vishid_w_target", rec_vishid_w_target))
updownsummaries.append(tf.summary.histogram("rec_hidpen_w_target", rec_hidpen_w_target))
updownsummaries.append(tf.summary.histogram("gen_hidpen_w_target", gen_hidpen_w_target))
updownsummaries.append(tf.summary.histogram("gen_vishid_w_target", gen_vishid_w_target))

updownsummaries.append(tf.summary.histogram("rec_vishid_b_target", rec_vishid_b_target))
updownsummaries.append(tf.summary.histogram("rec_hidpen_b_target", rec_hidpen_b_target))
updownsummaries.append(tf.summary.histogram("gen_hidpen_b_target", gen_hidpen_b_target))
updownsummaries.append(tf.summary.histogram("gen_vishid_b_target", gen_vishid_b_target))

updownsummaries.append(tf.summary.histogram("pentop_w_target", pentop_w_target))
updownsummaries.append(tf.summary.histogram("labtop_w_target", labtop_w_target))

#updown_input_image = tf.reshape(tf.transpose(vis), [-1, side, side, 1])
#updownsummaries.append(tf.summary.image("updown_input_image", updown_input_image, max_outputs=50))

tf.variables_initializer(updown_var_list).run(session=sess)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_writer = tf.summary.FileWriter(log + "/updown/", )
updownmerged = tf.summary.merge(updownsummaries)
print("performing up down algorithm")
for batch in tqdm(range(batch_count)):
    data, labels = mnist.train.next_batch(1)
    updates = [wakehid, targets, updownmerged, rec_vishid_w_target, rec_vishid_b_target, rec_hidpen_w_target, rec_hidpen_b_target, pentop_w_target, labtop_w_target, toppen_b_target, toplab_b_target, top_b_target, neglabprobs, gen_hidpen_w_target, gen_hidpen_b_target, gen_vishid_w_target, gen_vishid_b_target]
    w, t, summary, _, _, _, _, _, _, _, _, _, neg, _, _, _, _ = sess.run(updates, feed_dict={pentop_pixels:data, lab_on_deck: labels})
    #print("input to updown: " + str(t))
    #print("Neg lab probs: " + str(neg))
    #print("wakehid: " + str(w))
    train_writer.add_summary(summary, batch)

train_writer.close()

################################################################################################################ test

#
#l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#one_hot_labels = tf.cast(tf.nn.embedding_lookup(np.identity(max(l)+1), l), 'float')
#test_var.append(one_hot_labels)
#
#expansion_tensor = tf.ones([1, 10], 'float')
#test_var.append(expansion_tensor)
#wakepen10 = tf.matmul(wakepen, expansion_tensor)
##pentop_w10 = tf.matmul(pentop_w, expansion_tensor)
##labtop_w10 = tf.matmul(labtop_w, expansion_tensor)
#pentop_b10 = tf.matmul(pentop_b, expansion_tensor)
#
#topprobs = tf.sigmoid(tf.matmul(pentop_w, wakepen10) + tf.matmul(labtop_w, one_hot_labels) + pentop_b10)
#
#energies = (tf.matmul(topprobs, one_hot_labels) * labtop_w) + (pentop_w * tf.matmul(topprobs, wakepen10)) + (topprobs * pentop_b10)
#
#labprobs = tf.transpose(tf.nn.softmax(tf.reduce_sum(energies, 0)))
#
#print("labprobs shape = " + labprobs.shape)
#print("targets shape = " + targets.shape)

test_var = []

answers = tf.placeholder('float', [1, numlab], name='answers')

neglab_probs_target = neglabprobs

answer = tf.argmax(targets, 0)
prediction = tf.argmax(neglab_probs_target, 0)

correct_prediction = tf.equal(answer, prediction) 

accuracy = tf.cast(correct_prediction, tf.float32)

#tf.variables_initializer(test_var).run(session=sess)


#    tf.global_variables_initializer().run()
#    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#batch_count = 50
results = 0
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("testing network:")
for batch in range(batch_count):
    data, labels = mnist.train.next_batch(1)
    #data, labels = mnist.test.next_batch(1)
    goals = [accuracy, answer, prediction, neglab_probs_target]
    acc, ans, pred, probs = sess.run(goals, feed_dict={pentop_pixels:data})
    results += acc
    print("Neg lab probs: " + str(probs))
    print("Answer: " + str(ans) + ", Prediction: " + str(pred))

print("Accuracy: " + str((results/batch_count)*100) + "%")
        

sess.close()


