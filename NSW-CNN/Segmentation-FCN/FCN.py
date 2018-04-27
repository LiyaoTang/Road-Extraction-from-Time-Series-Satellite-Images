# encoding: utf-8

# setting environments
import sys
import os

from optparse import OptionParser

parser = OptionParser()
parser.add_option("--save", dest="save_path")
parser.add_option("--name", dest="model_name")
parser.add_option("--record_summary", action="store_true", default=False, dest="record_summary")

parser.add_option("--train", dest="path_train_set", default="../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_seg_coord_split_thr1_128_16_train")
parser.add_option("--cv", dest="path_cv_set", default="../../Data/090085/Road_Data/motor_trunk_pri_sec_tert_uncl_track/posneg_seg_coord_split_thr1_128_16_cv")

parser.add_option("--norm", default="mean", dest="norm")
parser.add_option("--pos", type="int", default=0, dest="pos_num")
parser.add_option("--size", type="int", default=128, dest="size")
parser.add_option("-e", "--epoch", type="int", default=15, dest="epoch")
parser.add_option("--learning_rate", type="float", default=9e-6, dest="learning_rate")
parser.add_option("--batch", type="int", default=1, dest="batch_size")
parser.add_option("--rand", type="int", dest="rand_seed")

parser.add_option("--conv", dest="conv_struct")
parser.add_option("--concat_input", dest="concat_input")
parser.add_option("--not_weight", action="store_false", default=True, dest="use_weight")
parser.add_option("--xavier_deconv", type="int", default=0, dest="xavier_deconv")
parser.add_option("--use_batch_norm", action="store_true", default=False, dest="use_batch_norm")
parser.add_option("--output_conv", type="int", default=3, dest="output_conv")

parser.add_option("--gpu", default="", dest="gpu")
parser.add_option("--gpu_max_mem", type="float", default=0.9, dest="gpu_max_mem")

(options, args) = parser.parse_args()

path_train_set = options.path_train_set
path_cv_set    = options.path_cv_set

save_path = options.save_path
model_name = options.model_name
record_summary = options.record_summary

norm = options.norm
pos_num = options.pos_num
size = options.size
epoch = options.epoch
batch_size = options.batch_size
learning_rate = options.learning_rate
rand_seed = options.rand_seed

conv_struct = options.conv_struct

xavier_deconv = options.xavier_deconv
use_weight = options.use_weight
use_batch_norm = options.use_batch_norm
concat_input = options.concat_input
output_conv = options.output_conv

gpu = options.gpu
gpu_max_mem = options.gpu_max_mem

# restrict to single gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

assert xavier_deconv in set([0,1])
xavier_deconv = (xavier_deconv == 1)

if norm.startswith('m'): norm = 'mean'
elif norm.startswith('G'): norm = 'Gaussian'
else: 
    print("norm = ", norm, " not in ('mean', 'Gaussian')")
    sys.exit()

if not model_name:
    model_name = "FCN_"
    model_name += conv_struct + "_" + str(output_conv) + "_"
    if concat_input: model_name += "cat" + concat_input + "_"
    if use_weight: model_name += "weight_"
    if use_batch_norm: model_name += "bn_"
    model_name += norm[0] + "_"
    model_name += "x" + str(options.xavier_deconv) + "_"
    model_name += "p" + str(pos_num) + "_"
    model_name += "e" + str(epoch) + "_"
    model_name += "r" + str(rand_seed)
    
if not save_path:
    print("no save path provided")
    sys.exit()
save_path = save_path.strip('/') + '/' + model_name + '/'

os.makedirs(save_path, exist_ok=True)
os.makedirs(save_path+'Analysis', exist_ok=True)

print("Train set:", path_train_set)
print("CV set:", path_cv_set)

print("will be saved as ", model_name)
print("will be saved into ", save_path)

if not conv_struct:
    print("must provide structure for conv")
    sys.exit()
else:
    conv_struct = [int(x) for x in conv_struct.split('-')]
    assert len(conv_struct) == 3

# parse concat_input options (if not None): e.g. 3-16;5-8;1-32 
# => concat[ 3x3 out_channel=16, 5x5 out_channel=8, 1x1 out_channel=32] followed by 1x1 conv out_channel = classoutput
# concat_input = 0 => concat the raw input before the calculation of logits
if concat_input:
    concat_input = [[int(x) for x in config.split('-')] for config in concat_input.split(';')]

# import libraries
import numpy as np
import tensorflow as tf
import sklearn.metrics as skmt
import matplotlib
matplotlib.use('agg') # so that plt works in command line
import matplotlib.pyplot as plt
import scipy.io as sio
import skimage.io
import h5py
import gc
import psutil
sys.path.append('../Metric/')
sys.path.append('../../Visualization/')
sys.path.append('../../Data_Preprocessing/')
from Metric import *
from Visualization import *
from Data_Extractor import *
from Bilinear_Kernel import *

# monitor mem usage
process = psutil.Process(os.getpid())
print('mem usage before data loaded:', process.memory_info().rss / 1024/1024, 'MB')
print()



''' Data preparation '''



# set random seed
if not (rand_seed is None):
    np.random.seed(rand_seed)

# Load training set
train_set = h5py.File(path_train_set, 'r')
train_pos_topleft_coord = np.array(train_set['positive_example'])
train_neg_topleft_coord = np.array(train_set['negative_example'])
train_raw_image = np.array(train_set['raw_image'])
train_road_mask = np.array(train_set['road_mask'])
train_set.close()

# Load cross-validation set
CV_set = h5py.File(path_cv_set, 'r')
CV_pos_topleft_coord = np.array(CV_set['positive_example'])
CV_neg_topleft_coord = np.array(CV_set['negative_example'])
CV_raw_image = np.array(CV_set['raw_image'])
CV_road_mask = np.array(CV_set['road_mask'])
CV_set.close()

Train_Data = FCN_Data_Extractor (train_raw_image, train_road_mask, size,
                                 pos_topleft_coord = train_pos_topleft_coord,
                                 neg_topleft_coord = train_neg_topleft_coord,
                                 normalization = norm)

CV_Data = FCN_Data_Extractor (CV_raw_image, CV_road_mask, size,
                              pos_topleft_coord = CV_pos_topleft_coord,
                              neg_topleft_coord = CV_neg_topleft_coord,
                              normalization = norm)
# run garbage collector
gc.collect()

print("train data:", train_raw_image.shape, train_road_mask.shape)
print("cv data:", CV_raw_image.shape, CV_road_mask.shape)

# monitor mem usage
process = psutil.Process(os.getpid())
print('mem usage after data loaded:', process.memory_info().rss / 1024/1024, 'MB')
print()



''' Create model '''



# general model parameter
band = 7

class_output = 2 # number of possible classifications for the problem
if use_weight:
    class_weight = [Train_Data.pos_size/Train_Data.size, Train_Data.neg_size/Train_Data.size]
    print(class_weight, '[neg, pos]')

iteration = int(Train_Data.size/batch_size) + 1

tf.reset_default_graph()
with tf.variable_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, size, size, band], name='x')
    y = tf.placeholder(tf.float32, shape=[None, size, size, class_output], name='y')
    weight      = tf.placeholder(tf.float32, shape=[None, size, size], name='weight')
    is_training = tf.placeholder(tf.bool, name='is_training') # batch norm

if use_batch_norm:
    normalizer_fn=tf.contrib.layers.batch_norm
    normalizer_params={'scale':True, 'is_training':is_training}
else:
    normalizer_fn=None
    normalizer_params=None

if xavier_deconv:
    deconv_init = lambda factor, in_channel, out_channel: tf.contrib.layers.xavier_initializer()
else:
    deconv_init = lambda factor, in_channel, out_channel: tf.constant_initializer(get_bilinear_upsample_weights(factor, in_channel, out_channel))

with tf.variable_scope('input_bridge'):
    if concat_input:
        if concat_input == [[0]]:
            input_map = x
        else:
            input_map = tf.concat([tf.contrib.layers.conv2d(inputs=x, num_outputs=cfg[1], kernel_size=cfg[0], 
                                                            stride=1, padding='SAME', scope=str(cfg[0])+'-'+str(cfg[1])) 
                                   for cfg in concat_input], 
                                  axis=-1)


with tf.variable_scope('down_sampling'):
    # Convolutional Layer 1
    net = tf.contrib.layers.conv2d(inputs=x, num_outputs=conv_struct[0], kernel_size=3, 
                                   stride=1, padding='SAME', normalizer_fn=normalizer_fn, normalizer_params=normalizer_params, scope='conv1')

    conv1 = net
    net = tf.contrib.layers.max_pool2d(inputs=net, kernel_size=2, stride=2, padding='VALID', scope='pool1')
    
    # Convolutional Layer 2
    net = tf.contrib.layers.conv2d(inputs=net, num_outputs=conv_struct[1], kernel_size=3, 
                                   stride=1, padding='SAME', normalizer_fn=normalizer_fn, normalizer_params=normalizer_params, scope='conv2')
    conv2 = net
    net = tf.contrib.layers.max_pool2d(inputs=net, kernel_size=2, stride=2, padding='VALID', scope='pool2')
    
    # # Convolutional Layer 3
    # net = tf.contrib.layers.conv2d(inputs=net, num_outputs=conv_struct[2], kernel_size=3, 
    #                                stride=1, padding='SAME', normalizer_fn=normalizer_fn, normalizer_params=normalizer_params, scope='conv3')
    # conv3 = net
    # net = tf.contrib.layers.max_pool2d(inputs=net, kernel_size=2, stride=2, padding='VALID', scope='pool3')


net = tf.contrib.layers.conv2d(inputs=net, num_outputs=conv_struct[-1], kernel_size=3, 
                               stride=1, padding='SAME', normalizer_fn=normalizer_fn, normalizer_params=normalizer_params, scope='bridge')


with tf.variable_scope('up_sampling'):
    kernel_size = get_kernel_size(2)
    # net = tf.contrib.layers.conv2d_transpose(inputs=net, num_outputs=conv_struct[2], kernel_size=kernel_size, stride=2, 
    #                                          weights_initializer=tf.constant_initializer(get_bilinear_upsample_weights(2, conv_struct[3], conv_struct[2])), 
    #                                          normalizer_fn=normalizer_fn, normalizer_params=normalizer_params, scope='conv3_T')
    # with tf.variable_scope('concat3'):
    #     net = tf.concat([net, conv3], axis=-1)

    net = tf.contrib.layers.conv2d_transpose(inputs=net, num_outputs=conv_struct[1], kernel_size=kernel_size, stride=2, 
                                             weights_initializer=deconv_init( 2, conv_struct[2], conv_struct[1]),
                                             normalizer_fn=normalizer_fn, normalizer_params=normalizer_params, scope='conv2_T')
    with tf.variable_scope('concat2'):
        net = tf.concat([net, conv2], axis=-1)
    
    net = tf.contrib.layers.conv2d_transpose(inputs=net, num_outputs=conv_struct[0], kernel_size=kernel_size, stride=2, 
                                             weights_initializer=deconv_init(2, conv_struct[1], conv_struct[0]), 
                                             normalizer_fn=normalizer_fn, normalizer_params=normalizer_params, scope='conv1_T')

    with tf.variable_scope('concat1'):
        net = tf.concat([net, conv1], axis=-1)

        if concat_input:
            with tf.variable_scope('concat_input'):
                net = tf.concat([net, input_map], axis=-1)

logits = tf.contrib.layers.conv2d(inputs=net, num_outputs=class_output, kernel_size=output_conv, stride=1, padding='SAME', scope='logits')

with tf.variable_scope('prob_out'):
    prob_out = tf.nn.softmax(logits, name='prob_out')

with tf.variable_scope('cross_entropy'):
    flat_logits = tf.reshape(logits, (-1, class_output), name='flat_logits')
    flat_labels = tf.to_float(tf.reshape(y, (-1, class_output)), name='flat_labels')
    flat_weight = tf.reshape(weight, [-1], name='flat_weight')

    cross_entropy = tf.losses.softmax_cross_entropy(flat_labels, flat_logits, weights=flat_weight)

# Ensures that we execute the update_ops before performing the train_step
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

if record_summary:            
    with tf.variable_scope('summary'):
        graph = tf.get_default_graph()

        # conv layers params
        conv_scopes = ['down_sampling/conv1', 'down_sampling/conv2', 'bridge', # 'down_sampling/conv3'
                       'up_sampling/conv2_T', 'up_sampling/conv1_T'] # 'up_sampling/conv3_T',

        for scope_name in conv_scopes:
            target_tensors = ['/weights:0']
            if use_batch_norm: target_tensors.extend(['/BatchNorm/gamma:0', '/BatchNorm/beta:0'])
            else: target_tensors.append('/biases:0')
            for tensor_name in target_tensors:
                tensor_name = scope_name + tensor_name
                cur_tensor = graph.get_tensor_by_name(tensor_name)
                tensor_name = tensor_name.split(':')[0]
                tf.summary.histogram(tensor_name, cur_tensor)
                tf.summary.histogram('grad_'+tensor_name, tf.gradients(cross_entropy, [cur_tensor])[0])

        # logits layer params
        scope_name = 'logits'
        target_tensors = ['/weights:0', '/biases:0']
        for tensor_name in target_tensors:
            tensor_name = scope_name + tensor_name
            cur_tensor = graph.get_tensor_by_name(tensor_name)
            tensor_name = tensor_name.split(':')[0]
            tf.summary.histogram(tensor_name, cur_tensor)
            tf.summary.histogram('grad_'+tensor_name, tf.gradients(cross_entropy, [cur_tensor])[0])

        # output layer
        tf.summary.image('input', tf.reverse(x[:,:,:,1:4], axis=[-1])) # axis must be of rank 1
        tf.summary.image('label', tf.expand_dims(y[:,:,:,1], axis=-1))
        tf.summary.image('prob_out_pos', tf.expand_dims(prob_out[:,:,:,1], axis=-1))
        tf.summary.image('prob_out_neg', tf.expand_dims(prob_out[:,:,:,0], axis=-1))
        tf.summary.image('logits_pos', tf.expand_dims(logits[:,:,:,1], axis=-1))
        tf.summary.image('logits_neg', tf.expand_dims(logits[:,:,:,0], axis=-1))
        tf.summary.scalar('cross_entropy', cross_entropy)
    merged_summary = tf.summary.merge_all()


# monitor mem usage
process = psutil.Process(os.getpid())
print('mem usage after model created:', process.memory_info().rss / 1024/1024, 'MB')
print()
sys.stdout.flush()



''' Train & monitor '''



saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_max_mem
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

balanced_acc_curve = []
AUC_curve = []
avg_precision_curve = []
cross_entropy_curve = []
if record_summary: train_writer = tf.summary.FileWriter('./Summary/FCN/' + model_name, sess.graph)
for epoch_num in range(epoch):
    for iter_num in range(iteration):

        batch_x, batch_y, batch_w = Train_Data.get_patches(batch_size=batch_size, positive_num=pos_num, norm=True, weighted=use_weight)
        batch_x = batch_x.transpose((0, 2, 3, 1))

        train_step.run(feed_dict={x: batch_x, y: batch_y, weight: batch_w, is_training: True})

    if record_summary:
        # tensor board
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary = sess.run(merged_summary, feed_dict={x: batch_x, y: batch_y, weight: batch_w, is_training: False}, options=run_options, run_metadata=run_metadata)

        train_writer.add_run_metadata(run_metadata, 'epoch_%03d' % (epoch_num+1))
        train_writer.add_summary(summary, epoch_num+1)

    # snap shot on CV set
    cv_metric = Metric_Record()
    cv_cross_entropy_list = []
    for batch_x, batch_y, batch_w in CV_Data.iterate_data(norm=True, weighted=use_weight):
        batch_x = batch_x.transpose((0, 2, 3, 1))

        [pred_prob, cross_entropy_cost] = sess.run([prob_out, cross_entropy], feed_dict={x: batch_x, y: batch_y, weight: batch_w, is_training: False})

        cv_metric.accumulate(Y         = np.array(batch_y.reshape(-1,class_output)[:,1]>0.5, dtype=int), 
                             pred      = np.array(pred_prob.reshape(-1,class_output)[:,1]>0.5, dtype=int), 
                             pred_prob = pred_prob.reshape(-1,class_output)[:,1])
        cv_cross_entropy_list.append(cross_entropy_cost)

    # calculate value
    balanced_acc = cv_metric.get_balanced_acc()
    AUC_score = skmt.roc_auc_score(np.array(cv_metric.y_true).flatten(), np.array(cv_metric.pred_prob).flatten())
    avg_precision_score = skmt.average_precision_score(np.array(cv_metric.y_true).flatten(), np.array(cv_metric.pred_prob).flatten())
    mean_cross_entropy = sum(cv_cross_entropy_list)/len(cv_cross_entropy_list)

    balanced_acc_curve.append(balanced_acc)
    AUC_curve.append(AUC_score)
    avg_precision_curve.append(avg_precision_score)
    cross_entropy_curve.append(mean_cross_entropy)

    print("mean_cross_entropy = ", mean_cross_entropy, "balanced_acc = ", balanced_acc, "AUC = ", AUC_score, "avg_precision = ", avg_precision_score)
    sys.stdout.flush()
print("finish")

# monitor mem usage
process = psutil.Process(os.getpid())
print('mem usage after model trained:', process.memory_info().rss / 1024/1024, 'MB')
print()

# plot training curve
plt.figsize=(9,5)
plt.plot(balanced_acc_curve, label='balanced_acc')
plt.plot(AUC_curve, label='AUC')
plt.plot(avg_precision_curve, label='avg_precision')
plt.legend()
plt.title('learning_curve_on_cross_validation')
plt.savefig(save_path+'Analysis/'+'cv_learning_curve.png', bbox_inches='tight')
plt.close()

plt.figsize=(9,5)
plt.plot(cross_entropy_curve)
plt.title('cv_cross_entropy_curve')
plt.savefig(save_path+'Analysis/'+'cv_cross_entropy_curve.png', bbox_inches='tight')
plt.close()

# save model
saver.save(sess, save_path + model_name)

# run garbage collection
saved_sk_obj = 0
gc.collect()



''' Evaluate model '''



# train set eva
print("On training set: ")
train_metric = Metric_Record()
train_cross_entropy_list = []
for batch_x, batch_y, batch_w in CV_Data.iterate_data(norm=True, weighted=use_weight):
    batch_x = batch_x.transpose((0, 2, 3, 1))

    [pred_prob, cross_entropy_cost] = sess.run([prob_out, cross_entropy], feed_dict={x: batch_x, y: batch_y, weight: batch_w, is_training: False})

    train_metric.accumulate(Y         = np.array(batch_y.reshape(-1,class_output)[:,1]>0.5, dtype=int),
                            pred      = np.array(pred_prob.reshape(-1,class_output)[:,1]>0.5, dtype=int), 
                            pred_prob = pred_prob.reshape(-1,class_output)[:,1])    
    train_cross_entropy_list.append(cross_entropy_cost)

train_metric.print_info()
AUC_score = skmt.roc_auc_score(np.array(train_metric.y_true).flatten(), np.array(train_metric.pred_prob).flatten())
avg_precision_score = skmt.average_precision_score(np.array(train_metric.y_true).flatten(), np.array(train_metric.pred_prob).flatten())
mean_cross_entropy = sum(train_cross_entropy_list)/len(train_cross_entropy_list)
print("mean_cross_entropy = ", mean_cross_entropy, "balanced_acc = ", balanced_acc, "AUC = ", AUC_score, "avg_precision = ", avg_precision_score)

# plot ROC curve
fpr, tpr, thr = skmt.roc_curve(np.array(train_metric.y_true).flatten(), np.array(train_metric.pred_prob).flatten())
plt.plot(fpr, tpr)
plt.savefig(save_path+'Analysis/'+'train_ROC_curve.png', bbox_inches='tight')
plt.close()

# cross validation eva
print("On CV set:")
cv_metric.print_info()

# plot ROC curve
fpr, tpr, thr = skmt.roc_curve(np.array(cv_metric.y_true).flatten(), np.array(cv_metric.pred_prob).flatten())
plt.plot(fpr, tpr)
plt.savefig(save_path+'Analysis/'+'cv_ROC_curve.png', bbox_inches='tight')
plt.close()
sys.stdout.flush()

# run garbage collection
train_metric = 0
cv_metric = 0
gc.collect()

# Predict road mask
# Predict road prob masks on train
train_pred_road = np.zeros([x for x in train_road_mask.shape] + [2])
for coord, patch in Train_Data.iterate_raw_image_patches_with_coord(norm=True):
    patch = patch.transpose((0, 2, 3, 1))
    train_pred_road[coord[0]:coord[0]+size, coord[1]:coord[1]+size, :] += logits.eval(feed_dict={x: patch, is_training: False})[0]

# Predict road prob on CV
CV_pred_road = np.zeros([x for x in CV_road_mask.shape] + [2])
for coord, patch in CV_Data.iterate_raw_image_patches_with_coord(norm=True):
    patch = patch.transpose((0, 2, 3, 1))
    CV_pred_road[coord[0]:coord[0]+size, coord[1]:coord[1]+size, :] += logits.eval(feed_dict={x: patch, is_training: False})[0]

# save prediction
prediction_name = model_name + '_pred.h5'
h5f_file = h5py.File(save_path + prediction_name, 'w')
h5f_file.create_dataset (name='train_pred', data=train_pred_road)
h5f_file.create_dataset (name='CV_pred', data=CV_pred_road)
h5f_file.close()

# monitor mem usage
process = psutil.Process(os.getpid())
print('mem usage after prediction maps calculated:', process.memory_info().rss / 1024/1024, 'MB')
print()