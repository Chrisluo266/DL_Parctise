import tensorflow as tf

class ClassifyModel(object):
    REGULAR_LOSS_KEY = "ws_regulars"

    def __init__(self,class_num,regularFactor=0,initLearningRate=0.001,activationfunc=None):
        self.class_num = class_num
        self.regularFactor = regularFactor
        self.activationfunc = activationfunc if activationfunc!=None else tf.nn.relu
        self.initLearningRate = initLearningRate

    def _get_weights(self,shape):
        weights = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
        if self.regularFactor < 0:
            tf.add_to_collection(ClassifyModel.REGULAR_LOSS_KEY,tf.contrib.layers.l1_regularizer(-self.regularFactor)(weights))
        elif self.regularFactor > 0:
            tf.add_to_collection(ClassifyModel.REGULAR_LOSS_KEY,tf.contrib.layers.l2_regularizer(self.regularFactor)(weights))
        return weights
    def _get_bians(self,shape):
        bias = tf.Variable(tf.constant(0.01,shape=shape),dtype=tf.float32)
        return bias

    def forward(self,input,dropout):
        input_dims = len(input.shape)
        if input_dims != 2:
            raise Exception("Error: Dim of Input. Expect is {}, but is {}".format(2,input_dims))
        l1_nums = self.class_num*5
        l1_weights = self._get_weights([int(input.shape[1]),l1_nums])
        l1_bians = self._get_bians([l1_nums])
        l1_re = self.activationfunc((tf.matmul(input,l1_weights) + l1_bians))
        l1_re = tf.nn.dropout(l1_re,dropout)

        l2_nums = self.class_num*3
        l2_weights = self._get_weights([l1_nums,l2_nums])
        l2_bians = self._get_bians([l2_nums])
        l2_re = self.activationfunc((tf.matmul(l1_re,l2_weights) + l2_bians))
        l2_re = tf.nn.dropout(l2_re,dropout)

        out_weights = self._get_weights([l2_nums,self.class_num])
        out_bians = self._get_bians([self.class_num])
        logits = tf.add(tf.matmul(l2_re,out_weights) ,out_bians,"logits")
        return logits

    def train(self,labels,logits,delay_step = 100,delay = 0.99):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels))
        if self.regularFactor != 0:
            loss = loss + tf.add_n(tf.get_collection(self.REGULAR_LOSS_KEY))
        
        global_step = tf.Variable(0,trainable=False)
        learningRate = tf.train.exponential_decay(self.initLearningRate,global_step,delay_step,delay,True)
        train_op = tf.train.AdamOptimizer(learningRate).minimize(loss,global_step=global_step)
        return train_op,loss,learningRate

    def accuracy(self,labels,logits):
        logits_index = tf.math.argmax(tf.nn.softmax(logits),1)
        labels_index = tf.math.argmax(labels,1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(logits_index,labels_index),dtype=tf.float32))
        return accuracy

    def models(self,delay_step,delay):
        x_placeholder = tf.placeholder(tf.float32,[None,2],"inputs")
        y_placeholder = tf.placeholder(tf.float32,[None,self.class_num],"labels")
        dropout_placeholder = tf.placeholder(tf.float32,name="dropout")

        logits = self.forward(x_placeholder,dropout_placeholder)
        train_op,loss,learningRate = self.train(y_placeholder,logits,delay_step,delay)

        #accuracy
        accuracy = self.accuracy(y_placeholder,logits)

        return train_op,logits,loss,accuracy,learningRate
        


