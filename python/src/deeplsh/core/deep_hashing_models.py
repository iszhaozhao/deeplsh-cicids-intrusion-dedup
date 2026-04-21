import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM, Dense, Lambda, Conv1D, GlobalMaxPooling1D, concatenate
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K
import scipy.stats as stats 
import itertools

from .similarities import *

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def _format_tqdm_logs(logs):
    if not logs:
        return {}
    formatted = {}
    for key, value in logs.items():
        try:
            formatted[key] = f"{float(value):.4f}"
        except (TypeError, ValueError):
            continue
    return formatted


class TqdmTrainingProgress(Callback):
    def __init__(self, desc="training"):
        super().__init__()
        self.desc = desc
        self.epoch_bar = None
        self.batch_bar = None

    def on_train_begin(self, logs=None):
        if tqdm is None:
            return
        total_epochs = self.params.get("epochs")
        self.epoch_bar = tqdm(total=total_epochs, desc=self.desc, unit="epoch", position=0, leave=True)

    def on_epoch_begin(self, epoch, logs=None):
        if tqdm is None:
            return
        total_steps = self.params.get("steps")
        self.batch_bar = tqdm(
            total=total_steps,
            desc=f"{self.desc} epoch {epoch + 1}",
            unit="batch",
            position=1,
            leave=False,
        )

    def on_train_batch_end(self, batch, logs=None):
        if self.batch_bar is None:
            return
        self.batch_bar.update(1)
        self.batch_bar.set_postfix(_format_tqdm_logs(logs), refresh=False)

    def on_epoch_end(self, epoch, logs=None):
        if self.batch_bar is not None:
            self.batch_bar.close()
            self.batch_bar = None
        if self.epoch_bar is not None:
            self.epoch_bar.update(1)
            self.epoch_bar.set_postfix(_format_tqdm_logs(logs), refresh=False)

    def on_train_end(self, logs=None):
        if self.batch_bar is not None:
            self.batch_bar.close()
            self.batch_bar = None
        if self.epoch_bar is not None:
            self.epoch_bar.close()
            self.epoch_bar = None

# 计算两个嵌入向量（Embedding）之间的改进型汉明距离（Hamming Distance），并将距离值归一化到 0~1 区间（值越大表示两个向量越相似）。
class HamDist(Layer):

    def __init__(self, b, m):
        self.b = b
        self.m = m
        self.result = None
        super(HamDist, self).__init__()

    def build(self, input_shape):
        super(HamDist, self).build(input_shape)

    def call(self, x):
        i = 0
        count = 0
        slicing = self.b
        size_embedding = self.b * self.m
        while i < size_embedding :
            count += K.max(K.abs(x[0][:,i:i+slicing] - x[1][:,i:i+slicing]), axis = 1) * slicing
            i = i + slicing
        self.result = 1 - count / (size_embedding * 2)
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

# 计算两个嵌入向量之间的改进型曼哈顿距离（Manhattan Distance），并通过指数变换将距离值映射为 0~1 区间的相似度值（值越大表示两个向量越相似）。
class ManhDist(Layer):

    def __init__(self, **kwargs):
        self.result = None
        super(ManhDist, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ManhDist, self).build(input_shape)

    def call(self, x, **kwargs):
        
        self.result = K.exp(-0.005*K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

# 对输入的嵌入向量（哈希向量）计算元素级平方和的均值，本质是对向量的 “能量 / 模长特征” 做归一化统计 
class ProdVec(Layer):

    def __init__(self, size_embedding):
        self.size_embedding = size_embedding
        self.result = None
        super(ProdVec, self).__init__()

    def build(self, input_shape):
        super(ProdVec, self).build(input_shape)

    def call(self, x):
        
        self.result = K.mean(K.sum((x * x), axis=1, keepdims=True) / self.size_embedding)
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

# 计算元素和的维度归一化值，输出每个样本的向量 “整体偏移特征”  
class SumVec(Layer):

    def __init__(self, size_embedding):
        self.size_embedding = size_embedding
        self.result = None
        super(SumVec, self).__init__()

    def build(self, input_shape):
        super(SumVec, self).build(input_shape)

    def call(self, x):
        
        self.result = K.sum(x, axis = 1) / self.size_embedding
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

# 对输入张量的每个元素取绝对值，无其他复杂计算逻辑  
class AbsVect(Layer):

    def __init__(self, **kwargs):
        self.result = None
        super(AbsVect, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AbsVect, self).build(input_shape)

    def call(self, x, **kwargs):
        
        self.result = K.abs(x)
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

# 计算输入张量（嵌入向量）与全 1 张量的余弦相似度，输出每个样本的余弦相似度值。    
class CosDist(Layer):

    def __init__(self, **kwargs):
        self.result = None
        super(CosDist, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CosDist, self).build(input_shape)

    def call(self, x, **kwargs):
        
        initializer = tf.keras.initializers.Ones()
        values_1 = initializer(shape=(128,1))
        self.result = K.dot(K.l2_normalize(K.abs(x), axis=-1), K.l2_normalize(values_1, axis=-1))
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
    
# 自定义损失函数，使用余弦相似度作为损失    
def custom_loss(y_true, y_pred):
    
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    return cosine_loss(y_true, y_pred)


def contrastive_similarity_loss(margin=0.7):
    def loss(y_true, y_pred):
        y_true = K.cast(K.reshape(y_true, (-1,)), K.floatx())
        y_pred = K.reshape(y_pred, (-1,))
        distance = 1.0 - y_pred
        positive_loss = y_true * 0.5 * K.square(distance)
        negative_loss = (1.0 - y_true) * 0.5 * K.square(K.maximum(0.0, margin - distance))
        return K.mean(positive_loss + negative_loss)

    loss.__name__ = "contrastive_discriminative_loss"
    return loss


def hash_balance_loss(hash_values, hash_activation="tanh"):
    bit_mean = K.mean(hash_values, axis=0)
    if hash_activation == "sigmoid":
        bit_mean = bit_mean - 0.5
    return K.mean(K.square(bit_mean))


def hash_quantization_loss(hash_values, hash_activation="tanh"):
    if hash_activation == "sigmoid":
        return K.mean(K.minimum(K.square(hash_values), K.square(1.0 - hash_values)))
    return K.mean(K.square(K.abs(hash_values) - 1.0))


def hash_regularization_loss(quantization_weight=0.01, balance_weight=0.1, hash_activation="tanh"):
    def loss(y_true, y_pred):
        quantization = hash_quantization_loss(y_pred, hash_activation=hash_activation)
        balance = hash_balance_loss(y_pred, hash_activation=hash_activation)
        return quantization_weight * quantization + balance_weight * balance

    return loss


# 构建孪生网络模型，包含哈希距离计算、能量统计和相似度计算等模块
def siamese_model(shared_model, input_shape, b, m, is_sparse = False, print_summary = True):
    size_hash_vector = m * b
    stack_1_input = Input(sparse = is_sparse, shape = input_shape)
    stack_2_input = Input(sparse = is_sparse, shape = input_shape)
    ham_distance = HamDist(b,m)([shared_model(stack_1_input), shared_model(stack_2_input)])
    model = Model(inputs = [stack_1_input, stack_2_input], outputs = [ham_distance,
                                                                      ProdVec(size_hash_vector)(shared_model(stack_1_input)), 
                                                                      ProdVec(size_hash_vector)(shared_model(stack_2_input)),
                                                                      SumVec(size_hash_vector)(shared_model(stack_1_input)),
                                                                      SumVec(size_hash_vector)(shared_model(stack_2_input))])

    metrics = [tf.keras.metrics.RootMeanSquaredError(name="rmse"), tf.keras.metrics.MeanAbsoluteError(name="mae")]
    model.compile(loss = ["mse", "mse", "mse","mse","mse"],
                  loss_weights=[15/16, 1/64, 1/64, 1/64, 1/64],
                  optimizer=tf.keras.optimizers.Adam(), 
                  metrics = [metrics, None, None, None, None])
    
    if print_summary : 
        print(model.summary())
        print(shared_model.summary())
    
    return model


def siamese_contrastive_model(
    shared_model,
    input_shape,
    b,
    m,
    margin=0.7,
    quantization_weight=None,
    balance_weight=None,
    alpha=1.0,
    beta=0.1,
    gamma=0.01,
    hash_activation="tanh",
    is_sparse=False,
    print_summary=True,
):
    if balance_weight is not None:
        beta = balance_weight
    if quantization_weight is not None:
        gamma = quantization_weight

    stack_1_input = Input(sparse=is_sparse, shape=input_shape)
    stack_2_input = Input(sparse=is_sparse, shape=input_shape)
    embedding_1 = shared_model(stack_1_input)
    embedding_2 = shared_model(stack_2_input)
    ham_similarity = HamDist(b, m)([embedding_1, embedding_2])
    model = Model(inputs=[stack_1_input, stack_2_input], outputs=ham_similarity)

    hash_values = K.concatenate([embedding_1, embedding_2], axis=0)
    balance = hash_balance_loss(hash_values, hash_activation=hash_activation)
    quantization = hash_quantization_loss(hash_values, hash_activation=hash_activation)
    quantization_weight_variable = tf.Variable(float(gamma), trainable=False, dtype=K.floatx(), name="quantization_weight")
    weighted_balance = beta * balance
    weighted_quantization = quantization_weight_variable * quantization
    model.add_loss(weighted_balance + weighted_quantization)
    model.add_metric(balance, name="balance_loss", aggregation="mean")
    model.add_metric(quantization, name="quantization_loss", aggregation="mean")
    model.add_metric(weighted_balance, name="weighted_balance_loss", aggregation="mean")
    model.add_metric(weighted_quantization, name="weighted_quantization_loss", aggregation="mean")
    model._quantization_weight_variable = quantization_weight_variable
    model._target_quantization_weight = float(gamma)

    metrics = [tf.keras.metrics.RootMeanSquaredError(name="rmse"), tf.keras.metrics.MeanAbsoluteError(name="mae")]
    model.compile(
        loss=contrastive_similarity_loss(margin=margin),
        loss_weights=[alpha],
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[*metrics, contrastive_similarity_loss(margin=margin)],
    )

    if print_summary:
        print(model.summary())
        print(shared_model.summary())

    return model


# 构建基线孪生网络模型，仅包含曼哈顿距离计算和能量统计模块
def siamese_model_baseline(shared_model, input_shape, is_sparse = False, print_summary = True):

    stack_1_input = Input(sparse = is_sparse, shape = input_shape)
    stack_2_input = Input(sparse = is_sparse, shape = input_shape)
    manh_distance = ManhDist()([shared_model(stack_1_input), shared_model(stack_2_input)])
    model = Model(inputs = [stack_1_input, stack_2_input], outputs = [manh_distance,
                                                                      AbsVect()(shared_model(stack_1_input)), 
                                                                      AbsVect()(shared_model(stack_2_input))])
    metrics = [tf.keras.metrics.RootMeanSquaredError(name="rmse"), tf.keras.metrics.MeanAbsoluteError(name="mae")]
    model.compile(loss = ['mse', custom_loss, custom_loss],
                  loss_weights=[1/2, 1/4, 1/4],
                  optimizer=tf.keras.optimizers.Adam(), 
                  metrics = [metrics, None, None])
    
    if print_summary : 
        print(model.summary())
        print(shared_model.summary())
    
    return model
        

# 训练孪生网络模型，使用训练数据和验证数据进行模型训练，并返回训练历史记录
def train_siamese_model(model, X_train, X_validation, Y_train, Y_validation, batch_size, epochs, progress_desc="training"):
    _1_train = np.ones((Y_train.size,))
    _1_validation = np.ones((Y_validation.size,))
    _0_train = np.zeros((Y_train.size,))
    _0_validation = np.zeros((Y_validation.size,))
    callbacks = [TqdmTrainingProgress(progress_desc)] if tqdm is not None else []
    fit_verbose = 0 if callbacks else 1
    siamese_model = model.fit([X_train['stack_1'], X_train['stack_2']], [Y_train, _1_train, _1_train, _0_train, _0_train],
                      batch_size = batch_size,
                      epochs = epochs,
                      validation_data=([X_validation['stack_1'], X_validation['stack_2']], [Y_validation, _1_validation, _1_validation, _0_validation, _0_validation]),
                      callbacks=callbacks,
                      verbose=fit_verbose)
    return siamese_model


def train_siamese_contrastive_model(model, X_train, X_validation, Y_train, Y_validation, size_hash_vector=None, batch_size=128, epochs=10, progress_desc="training"):
    return train_siamese_contrastive_model_with_warmup(
        model,
        X_train,
        X_validation,
        Y_train,
        Y_validation,
        size_hash_vector=size_hash_vector,
        batch_size=batch_size,
        epochs=epochs,
        quantization_warmup_epochs=0,
        progress_desc=progress_desc,
    )


def train_siamese_contrastive_model_with_warmup(
    model,
    X_train,
    X_validation,
    Y_train,
    Y_validation,
    size_hash_vector=None,
    batch_size=128,
    epochs=10,
    quantization_warmup_epochs=0,
    progress_desc="training",
):
    callbacks = []
    if tqdm is not None:
        callbacks.append(TqdmTrainingProgress(progress_desc))
    fit_verbose = 0 if callbacks else 1
    quantization_weight_variable = getattr(model, "_quantization_weight_variable", None)
    target_quantization_weight = float(getattr(model, "_target_quantization_weight", 0.0))
    if quantization_weight_variable is not None and quantization_warmup_epochs > 0:
        K.set_value(quantization_weight_variable, 0.0)

        class QuantizationWarmup(Callback):
            def on_epoch_begin(self, epoch, logs=None):
                progress = min(1.0, max(0.0, float(epoch) / float(quantization_warmup_epochs)))
                K.set_value(quantization_weight_variable, target_quantization_weight * progress)

            def on_train_end(self, logs=None):
                K.set_value(quantization_weight_variable, target_quantization_weight)

        callbacks.append(QuantizationWarmup())

    history = model.fit(
        [X_train['stack_1'], X_train['stack_2']],
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([X_validation['stack_1'], X_validation['stack_2']], Y_validation),
        callbacks=callbacks,
        verbose=fit_verbose,
    )
    return history

# 训练基线孪生网络模型，使用训练数据和验证数据进行模型训练，并返回训练历史记录
def train_siamese_model_baseline(model, X_train, X_validation, Y_train, Y_validation, size_hash_vector, batch_size, epochs, progress_desc="training"):
    _1_train = np.ones((Y_train.size, size_hash_vector))
    _1_validation = np.ones((Y_validation.size, size_hash_vector))
    callbacks = [TqdmTrainingProgress(progress_desc)] if tqdm is not None else []
    fit_verbose = 0 if callbacks else 1
    siamese_model = model.fit([X_train['stack_1'], X_train['stack_2']], [Y_train, _1_train, _1_train],
                      batch_size = batch_size,
                      epochs = epochs,
                      validation_data=([X_validation['stack_1'], X_validation['stack_2']], [Y_validation, _1_validation, _1_validation]),
                      callbacks=callbacks,
                      verbose=fit_verbose)
    return siamese_model


def predict_with_tqdm(model, data, batch_size=128, desc="predict"):
    if tqdm is None:
        return model.predict(data, batch_size=batch_size, verbose=1)

    n_rows = int(data.shape[0])
    output_dim = int(model.output_shape[-1])
    outputs = np.empty((n_rows, output_dim), dtype=np.float32)
    total_batches = int(np.ceil(n_rows / float(batch_size)))
    with tqdm(total=total_batches, desc=desc, unit="batch") as progress:
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            outputs[start:end] = model.predict(data[start:end], verbose=0)
            progress.update(1)
    return outputs


# 预测函数，使用训练好的模型对输入数据进行预测
def predict(model, X):
    return model.predict([X['stack_1'], X['stack_2']])[0].reshape(1,-1)


# 计算预测结果与真实值之间的斯皮尔曼相关系数
def spearman_rho(predictions, real_values):
    rho, p_value = stats.spearmanr(predictions[0], real_values)
    return (rho, p_value)

# 计算预测结果与真实值之间的肯德尔相关系数
def kendall_tau(predictions, real_values):
    tau, p_value = stats.kendalltau(predictions[0], real_values)
    return (tau, p_value)


# 将输入值转换为1或-1
def transform (x):
    return 1 if x > 0 else -1

# 计算两个嵌入向量之间的汉明距离
def hamming(embedding1, embedding2, slicing, length) :
    count = 0
    i = 0
    while i < length :
        if np.unique(embedding1[i:i+slicing] == embedding2[i:i+slicing]).shape[0] == 1 & np.unique(embedding1[i:i+slicing] == embedding2[i:i+slicing])[0] == True :
            count += 1
        i += slicing
    return count / length * slicing

# 计算两个嵌入向量之间的改进型汉明距离
def hamming_diff(embedding1, embedding2, slicing, length) :
    i = 0
    count = 0
    while i < length :

        count += np.max(np.abs(embedding1[i:i+slicing] - embedding2[i:i+slicing])) * slicing
        i += slicing
    return 1 - count / (length * 2)

        
# 获取训练好的模型的中间层输出
def intermediate_model_trained(shared_model, output_layer, CNN = False, input_tensor = None):
    if CNN :
        return Model(inputs = input_tensor, outputs = shared_model.layers[output_layer].output)
    else :
        return Model(inputs = shared_model.input, outputs = shared_model.layers[output_layer].output)  
    

# 比较两个嵌入向量之间的汉明距离
def compare_hamming(X, intermediate_model, b, size_embedding):
    
    df_hamming = pd.DataFrame()
    df_hamming['embedding_stack_1'] = pd.Series(intermediate_model.predict(X['stack_1']).tolist())
    df_hamming['embedding_stack_2'] = pd.Series(intermediate_model.predict(X['stack_2']).tolist())
    df_hamming['embedding_stack_1'] = df_hamming['embedding_stack_1'].apply(lambda x : np.array(list(map(transform, x))))
    df_hamming['embedding_stack_2'] = df_hamming['embedding_stack_2'].apply(lambda x : np.array(list(map(transform, x))))
    df_hamming['hamming'] = df_hamming.apply(lambda x : hamming(x['embedding_stack_1'], x['embedding_stack_2'], b, size_embedding), axis = 1)
    return df_hamming


# 将帧号转换为索引
def index_frame(l, df) :
    list_index = []
    for elt in l :
        try :
            list_index.append(df.index[df['frame'] == elt][0] + 1)
        except :
            list_index.append(0)
    return list_index

# 分配堆栈
def assign_stacks (index, df) :
    n = df.shape[0]
    a,b = get_indices_sim(n, index)
    return (df['rankFrames'][a], df['rankFrames'][b])

# 填充序列
def padding(df, max_seq_length):
    
    dict_X = {'stack_1': df['stack1'], 'stack_2': df['stack2']}

    for data, side in itertools.product([dict_X], ['stack_1', 'stack_2']):
        data[side] = pad_sequences(data[side], padding = 'post', truncating = 'post', maxlen = max_seq_length)

    return data
