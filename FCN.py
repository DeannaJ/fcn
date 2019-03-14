from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange
#six.moves 是用来处理那些在2 和 3里面函数的位置有变化的，直接用six.moves就可以屏蔽掉这些变化
#xrange 用来处理数据类型切换

#执行main函数之前首先进行flags的解析，也就是说TensorFlow通过设置flags来传递tf.app.run()所需要的参数，
#我们可以直接在程序运行前初始化flags，也可以在运行程序的时候设置命令行参数来达到传参的目的。
##调用flags内部的DEFINE_string函数来制定解析规则
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
#vlfeat开源计算机视觉库
#matconvnet是实现用于计算机视觉领域的卷积神经网络的matlab工具箱，包含很多CNN计算模块如卷积、归一化、池化等
#它可以学习AlexNet等大型深度CNN模型
MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 151
IMAGE_SIZE = 224

## vgg 网络部分， weights 是vgg网络各层的权重集合， image是被预测的图像的向量
def vgg_net(weights, image):
    ## fcn的前五层网络就是vgg网络
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    #vgg的每层结果保存在net中
    #卷积支持输入补0输出下采样；反卷积支持输入上采样输出裁剪
    net = {}
    current = image #输入图像
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            # 由于 imagenet-vgg-verydeep-19.mat 中的参数矩阵和我们定义的长宽位置颠倒了
            #原来索引号（reshape(2,2,3)）是012，现在是102
            #(1, 0, 2, 3)是索引号
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            print('kernels:',kernels)
            #reshape(-1)把bias参数数组合并成一行
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            print('bias:',bias)
            current = utils.conv2d_basic(current, kernels, bias)
            print('current:',current)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            print('current_relu:',current)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            ## vgg 的前5层的stride都是2，也就是前5层的size依次减小1倍
            ## 这里处理了前4层的stride，用的是平均池化
            ## 第5层的pool在下文的外部处理了，用的是最大池化
            ## pool1 size缩小2倍
            ## pool2 size缩小4倍
            ## pool3 size缩小8倍
            ## pool4 size缩小16倍
            current = utils.avg_pool_2x2(current)   ##平均池化
        net[name] = current
        ##vgg的每层结果都保存在net中
    return net

## 预测流程，image是输入图像的向量，keep_prob是dropout rate
## dropout是防止网络过拟合，深度学习网络训练的过程中，按照一定的概率将一部分网络单元暂时从网络中
## 放弃，相当于从原始的网络中找一个更瘦的网络
def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    ##语义分割网络定义
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    ##获取训练好的VGG部分的 model
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])
    #将图像的向量值都减去平均像素值，进行normalization
    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        #计算前5层vgg网络的输出结果
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]
        ## pool1 size缩小2倍
        ## pool2 size缩小4倍
        ## pool3 size缩小8倍
        ## pool4 size缩小16倍
        ## pool5 size缩小32倍
        pool5 = utils.max_pool_2x2(conv_final_layer)
        #初始化第六层的w,b
        #7*7卷积核视野很大
        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
        ## 在第6层没有进行池化，所以经过第6层后 size缩小仍为32倍
        
        # 初始化第7层的w,b
        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
        ## 在第7层没有进行池化，所以经过第7层后 size缩小仍为32倍
        
        ## 初始化第8层的w、b
        ## 输出维度为NUM_OF_CLASSESS=151
        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        # 开始将size提升为原始尺寸(反卷积)
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        ## 对第8层的结果进行反卷积(上采样),通道数也由NUM_OF_CLASSESS变为第4层的通道数
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        ## 对应论文原文中的"2× upsampled prediction + pool4 prediction"
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
        
        ## 对上一层上采样的结果进行反卷积(上采样),通道数也由上一层的通道数变为第3层的通道数
        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        ## 对应论文原文中的"2× upsampled prediction + pool3 prediction"
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")
         ## 原始图像的height、width和通道数
        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        # 再进行一次反卷积，将上一层的结果转化为和原始图像相同size、通道数为分类数的形式数据
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
        ## 目前conv_t3的形式为size为和原始图像相同的size，通道数与分类数相同  
        ## 这句我的理解是对于每个像素位置，根据3个维度（通道数即RGB的值）通过argmax能计算出这个像素点属于哪个分类  
        ## 也就是对于每个像素而言，NUM_OF_CLASSESS个通道中哪个数值最大，这个像素就属于哪个分类 
        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3

#定义训练损失优化器及训练的梯度下降方法以更新参数
def train(loss_val, var_list):  #测试损失
    #Adam 这个名字来源于adaptive moment estimation，自适应矩估计，
    #通常都会得到比SGD算法性能更差（经常是差很多）的结果，尽管自适应优化算法在训练时会表现的比较好，
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    ## 下面是参照tf api
    ## Compute gradients of loss_val for the variables in var_list.
    ## This is the first part of minimize().
    ## loss: A Tensor containing the value to minimize.
    ## var_list: Optional list of tf.Variable to update to minimize loss.
    ##   Defaults to the list of variables collected in the graph under the key GraphKey.TRAINABLE_VARIABLES.
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
     ## 下面是参照tf api
    ## Apply gradients to variables.
    ## This is the second part of minimize(). 
    #   It returns an Operation that applies gradients.
    return optimizer.apply_gradients(grads)


def main(argv=None):
    #dropout的保留率
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    #原始图像的向量
    #placeholder可以理解为形参，用于定义过程，在执行的时候再赋值
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    ## 原始图像对应的标注图像的向量
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    ## 为了方便查看图像预处理的效果，可以利用 TensorFlow 提供的 tensorboard 工具进行可视化，
    #直接用 tf.summary.image 将图像写入 summary
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    ### 计算预测标注图像和真实标注图像的交叉熵
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    loss_summary = tf.summary.scalar("entropy", loss)
    ## 返回需要训练的变量列表
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    # create two summary writers to show training loss and validation loss in the same graph
    # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                train_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss, summary_sva = sess.run([loss, loss_summary], feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                # add validation loss to TensorBoard
                validation_writer.add_summary(summary_sva, itr)
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)

#使用这种方式保证了如果该文件被其他文件import的时候，不会执行main函数
if __name__ == "__main__":
    tf.app.run()    #解析命令行参数，调用main函数
