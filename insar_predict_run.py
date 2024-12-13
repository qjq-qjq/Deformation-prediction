import datetime
import os.path
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import sklearn.preprocessing as pre
import Model_Collection_Class as model_collection
import Model_Config as model_config_dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import rcParams

def primes(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return np.nonzero(sieve)[0]

def good_point_set(M, N):
    k = np.arange(1, M + 1).reshape(-1, 1) * np.ones((1, N))
    Ind = np.arange(1, N + 1)
    prime1 = np.array(list(primes(100 * N)))
    q = np.where(prime1 >= (2 * N + 3))[0]
    p = prime1[q[0]]
    tmp2 = 2 * np.cos((2 * np.pi * Ind) / p)
    r = np.ones((M, 1)) * tmp2
    G_p = k * r
    Good_points = np.mod(G_p, 1)
    return Good_points


def gps_initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = len(ub)  # number of boundaries

    # If the boundaries of all variables are equal and user enter a single number for both ub and lb
    if Boundary_no == 1:
        Positions = good_point_set(SearchAgents_no, dim) * (ub - lb) + lb
    # If each variable has a different lb and ub
    elif Boundary_no > 1:
        Good_points = good_point_set(SearchAgents_no, dim)
        Positions = np.zeros((SearchAgents_no, dim))
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = Good_points[:, i] * (ub_i - lb_i) + lb_i
    return Positions



def objective_function(model_type, x, train_loader, test_loader_list, epochs, pre_feature_scaler, input_size,
                       predict_date_list):
    learning_rate = x[0]
    if model_type == 'GVSAO-LSTM-TCN':

        KerlSize = int(np.round(x[3]))

        hidden_size = int(np.round(x[2]))
        num_layers = int(np.round(x[1]))
        output_size = model_config_dict.LSTM_TCN_Param['output_size']
        kernel_size = KerlSize
        dropout = model_config_dict.LSTM_TCN_Param['dropout']

        if len(x) == 6:
            tcn_layers = int(np.round(x[4]))
            tcn_hidden_size = int(np.round(x[5]))
            num_channels = [tcn_hidden_size] * tcn_layers
        else:
            num_channels = model_config_dict.LSTM_TCN_Param['num_channels']

    model = model_collection.LSTM_TCN(input_size, hidden_size, num_layers, output_size, num_channels, kernel_size,
                                      dropout)

    # ------模型训练--------
    model, epoch_list, loss_list = model_train(model, learning_rate, epochs, train_loader)
    # -----模型测试----
    eval_data_dict = predict_model_eval(test_loader_list, model, pre_feature_scaler, predict_date_list)
    # 模型评估
    all_real_y_array = eval_data_dict['real'].values.flatten()
    all_pre_y_array = eval_data_dict['predict'].values.flatten()

    filter_index = np.where(abs(all_real_y_array) < 0.1)
    filter_all_real_y_array = np.delete(all_real_y_array, filter_index)
    filter_all_pre_y_array = np.delete(all_pre_y_array, filter_index)
    index_result = model_eval_index(real_y=filter_all_real_y_array, pred_y=filter_all_pre_y_array)
    R = index_result['MAPE']


    object_data_model = {
        'model': model,
        'epochs': epoch_list,
        'loss': loss_list,
        'eval': eval_data_dict,
        'R': R,
        'index_result': index_result
    }
    return object_data_model


def GVSAO_optim_model(SearchAgents_no, Max_iter, lb, ub, model_type, train_loader, test_loader, epochs,
                      pre_feature_scaler, input_size, predict_date_list, out_param):
    dim = len(lb)
    X = gps_initialization(SearchAgents_no, dim, ub, lb)



    if model_type == 'GVSAO-LSTM-TCN':
        learning_rate = model_config_dict.LSTM_TCN_Param['learning_rate']
        num_layers = model_config_dict.LSTM_TCN_Param['num_layers']
        hidden_size = model_config_dict.LSTM_TCN_Param['hidden_size']
        kernel_size = model_config_dict.LSTM_TCN_Param['kernel_size']
        num_channels = model_config_dict.LSTM_TCN_Param['num_channels']
        tcn_layers = len(num_channels)
        tcn_hidden_size = num_channels[-1]
        if len(lb) == 6:
            add_x = [learning_rate, num_layers, hidden_size, kernel_size, tcn_layers, tcn_hidden_size]
        else:
            add_x = [learning_rate, num_layers, hidden_size, kernel_size]
        X[-1] = add_x

    Best_pos = np.zeros(dim)
    Best_score = float('inf')

    Objective_values = np.zeros(SearchAgents_no)
    Convergence_curve = []
    N1 = int(SearchAgents_no * 0.5)
    Elite_pool = []


    Am = 5
    fr = 10

    data_model_Info = {}
    for i in range(SearchAgents_no):

        object_data_model = objective_function(model_type, X[i, :], train_loader, test_loader, epochs,
                                               pre_feature_scaler, input_size, predict_date_list)


        Objective_values[i] = object_data_model['R']
        data_model_Info[i] = object_data_model
        if i == 0 or Objective_values[i] < Best_score:

            Best_pos = X[i, :].copy()
            Best_pos[1:] = np.int32(np.round(Best_pos[1:]))

            Best_score = Objective_values[i]

            Best_Info = data_model_Info[i]

    idx1 = np.argsort(Objective_values)
    Elite_pool.append(X[idx1[0], :])
    Elite_pool.append(X[idx1[1], :])
    Elite_pool.append(X[idx1[2], :])
    Elite_pool.append(np.mean(X[idx1[:N1], :], axis=0))

    Convergence_curve.append(Best_score)

    Na = Nb = SearchAgents_no // 2


    l = 1
    while l < Max_iter:
        RB = np.random.randn(SearchAgents_no, dim)
        T = np.exp(-l / Max_iter)
        k = 1
        DDF = 0.4 * (1 + (3 / 5) * (np.exp(l / Max_iter) - 1) ** k / (np.exp(1) - 1) ** k)
        M = DDF * T


        X_centroid = np.mean(X, axis=0)


        index1 = random.sample(list(range(SearchAgents_no)), Na)
        index2 = list(set(range(SearchAgents_no)) - set(index1))

        for i in index1:
            r1 = np.random.random()
            k1 = np.random.randint(0, 4)
            if (l + fr) % fr == 0:
                X[i, :] = X[i, :] * (1 + Am * (0.5 - np.random.rand()))
            else:
                X[i, :] = Elite_pool[k1] + RB[i] * (r1 * (Best_pos - X[i, :]) + (1 - r1) * (X_centroid - X[i, :]))

        if Na < SearchAgents_no:
            Na += 1
            Nb -= 1

        if Nb >= 1:
            for i in index2:
                r2 = 2 * np.random.random() - 1
                X[i, :] = M * Best_pos + RB[i] * (r2 * (Best_pos - X[i, :]) + (1 - r2) * (X_centroid - X[i, :]))


        X = np.clip(X, lb, ub)


        for i in range(SearchAgents_no):

            object_data_model = objective_function(model_type, X[i, :], train_loader, test_loader, epochs,
                                                   pre_feature_scaler, input_size, predict_date_list)


            Objective_values[i] = object_data_model['R']
            data_model_Info[i] = object_data_model
            if Objective_values[i] < Best_score:

                Best_pos = X[i, :].copy()
                Best_pos[1:] = np.int32(np.round(Best_pos[1:]))

                Best_score = Objective_values[i]

                Best_Info = data_model_Info[i]


        idx1 = np.argsort(Objective_values)
        Elite_pool[0] = X[idx1[0], :]
        Elite_pool[1] = X[idx1[1], :]
        Elite_pool[2] = X[idx1[2], :]
        Elite_pool[3] = np.mean(X[idx1[:N1], :], axis=0)

        Convergence_curve.append(Best_score)
        l += 1

    gvsao_result = {
        'Best_score': Best_score,
        'Best_pos': Best_pos,
        'Convergence_curve': Convergence_curve,
    }
    #
    gvsao_result.update(Best_Info)
    return gvsao_result


def in_out_sequence(data, window_size, y_dim):
    lg = len(data)
    train_x_list = []
    train_y_list = []
    for i in range(lg - window_size):
        train_x = data[i:i + window_size]
        train_y = data[i + window_size:i + window_size + 1, y_dim]
        train_x_list.append(train_x)
        train_y_list.append(train_y)
    return train_x_list, train_y_list


def data_loader_generator_bytime(insar_df, insar_list, features, window_size, train_batchsize):

    date_list = list(insar_df['date'].unique())
    train_num = int(len(date_list) * 0.8)
    train_date_list = date_list[:train_num + window_size]
    test_date_list = date_list[train_num:]

    all_train_df = insar_df[insar_df['date'].isin(train_date_list)]
    all_train_values = all_train_df[features].values

    all_test_df = insar_df[insar_df['date'].isin(test_date_list)]
    all_test_values = all_test_df[features].values


    input_feature_scaler = pre.MinMaxScaler(feature_range=(-1, 1))
    y_scaler = pre.MinMaxScaler(feature_range=(-1, 1))
    train_normalize = input_feature_scaler.fit_transform(all_train_values)
    test_normalize = input_feature_scaler.transform(all_test_values)
    y_scaler.fit(all_train_values[:, -1:])


    all_train_df = all_train_df.reset_index()
    train_x_list = []
    train_y_list = []
    for item in insar_list:
        one_insar_index = all_train_df.index[all_train_df['index'] == item]
        one_insar_array = train_normalize[one_insar_index.values, :]
        one_train_x_list, one_train_y_list = in_out_sequence(one_insar_array, window_size, -1)
        train_x_list = train_x_list + one_train_x_list
        train_y_list = train_y_list + one_train_y_list

    x_train_tensor = torch.FloatTensor(np.array(train_x_list))
    y_train_tensor = torch.FloatTensor(np.array(train_y_list))
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batchsize, shuffle=False)

    all_test_df = all_test_df.reset_index()
    test_loader_list = []

    for item in insar_list:
        one_insar_index = all_test_df.index[all_test_df['index'] == item]
        one_insar_array = test_normalize[one_insar_index.values, :]
        one_insar_values = all_test_values[one_insar_index.values, :]
        one_test_x_list, one_test_y_list = in_out_sequence(one_insar_array, window_size, y_dim=-1)

        x_test_tensor = torch.FloatTensor(np.array(one_test_x_list))
        y_test_tensor = torch.FloatTensor(np.array(one_test_y_list))
        test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        test_loader_list.append({
            'test_loader': test_loader,
            'insar_code': item,
            'real_values': one_insar_values
        })
    pre_num = len(test_date_list) - window_size
    dataset_dict = {
        'train': train_loader,
        'test': test_loader_list,
        'y_scaler': y_scaler,
        'predict_date_list': date_list[-pre_num:],
        'pre_num': pre_num
    }
    return dataset_dict


def model_Instancing(model_type, input_size):

    hidden_size = model_config_dict.LSTM_TCN_Param['hidden_size']
    num_layers = model_config_dict.LSTM_TCN_Param['num_layers']
    output_size = model_config_dict.LSTM_TCN_Param['output_size']
    kernel_size = model_config_dict.LSTM_TCN_Param['kernel_size']
    dropout = model_config_dict.LSTM_TCN_Param['dropout']
    num_channels = model_config_dict.LSTM_TCN_Param['num_channels']
    model = model_collection.LSTM_TCN(input_size, hidden_size, num_layers, output_size, num_channels, kernel_size,
                                      dropout)
    return model

def model_train(model, learning_rate, epochs, train_loader):

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_list = []
    epoch_list = []
    model.train()
    for epoch in range(1, epochs + 1):
        print('---epoch---', epoch)
        epoch_list.append(epoch)
        epoch_loss = 0
        for x, y in train_loader:
            outputs = model(x)
            optimizer.zero_grad()
            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss.item()
        loss_list.append(epoch_loss / len(train_loader))

    return model, epoch_list, loss_list



def model_eval_index(real_y, pred_y):

    MSE = mean_squared_error(real_y, pred_y)
    MAE = mean_absolute_error(real_y, pred_y)
    RMSE = np.sqrt(mean_squared_error(real_y, pred_y))
    MAPE = np.mean(np.abs((pred_y - real_y) / real_y))
    R2 = r2_score(real_y, pred_y)
    result = {
        'MSE': MSE,
        'MAE': MAE,
        'RMSE': RMSE,
        'MAPE': MAPE,
        'R2': R2
    }
    return result


def predict_model_eval(test_loader_list, model, pre_feature_scaler, predict_date_list):

    pre_y_list = []
    real_y_list = []
    # model.eval()
    with torch.no_grad():
        eval_data_dict = {}
        insar_list = []
        for test_item in test_loader_list:

            oneinsar_pre_y_list = []
            oneinsar_real_y_list = []
            insar_code = test_item['insar_code']
            test_loader = test_item['test_loader']
            real_values = test_item['real_values']
            for x, y in test_loader:
                prediction = model(x)

                oneinsar_pre_y_list.append(prediction.squeeze(0).detach().numpy())
                oneinsar_real_y_list.append(y.squeeze(0).detach().numpy())
            pred_y_array = np.array(oneinsar_pre_y_list)
            real_y_array = np.array(oneinsar_real_y_list)

            pred_data = pre_feature_scaler.inverse_transform(pred_y_array)
            real_data = pre_feature_scaler.inverse_transform(real_y_array)
            insar_list.append(insar_code)
            pre_y_list.append(pred_data.flatten())
            real_y_list.append(real_data.flatten())
        pre_y_df = pd.DataFrame(pre_y_list, index=insar_list, columns=predict_date_list)
        real_y_df = pd.DataFrame(real_y_list, index=insar_list, columns=predict_date_list)

    eval_data_dict['predict'] = pre_y_df
    eval_data_dict['real'] = real_y_df
    return eval_data_dict


def multi_predict_model_main(insar_df: pd.DataFrame, insar_list: list, features: list, out_param: dict):

    model_out_dir = out_param['model_out_dir']
    prefix_name = out_param['prefix_name']
    window_size = model_config_dict.Common_Param['window_size']
    train_batchsize = model_config_dict.Common_Param['train_batchsize']
    input_size = len(features)

    if 'freq' in features:
        freq_series = insar_df.groupby('index', as_index=False)['freq'].shift(periods=-1)
        insar_df.loc[:, 'freq'] = freq_series

    dataset_dict = data_loader_generator_bytime(insar_df, insar_list, features, window_size, train_batchsize)
    train_loader = dataset_dict['train']
    test_loader_list = dataset_dict['test']
    pre_feature_scaler = dataset_dict['y_scaler']
    predict_date_list = dataset_dict['predict_date_list']
    pre_num = dataset_dict['pre_num']

    epochs = model_config_dict.Common_Param['epochs']
    loss_data_list = []
    index_result_list = []
    eval_data_dict_list = []

    print('start model LSTM-TCN..........')
    model_type = 'LSTM-TCN'
    learning_rate = model_config_dict.LSTM_TCN_Param['learning_rate']
    lstm_tcn = model_Instancing(model_type=model_type, input_size=input_size)
    model, epoch_list, loss_list = model_train(lstm_tcn, learning_rate, epochs, train_loader)
    model_filepath = os.path.join(model_out_dir, prefix_name + '-' + model_type + '.pt')
    torch.save(model, model_filepath)
    loss_data_list.append({
        'type': model_type,
        'epochs': epoch_list,
        'loss': loss_list
    })

    eval_data_dict = predict_model_eval(test_loader_list, model, pre_feature_scaler, predict_date_list)
    eval_data_dict['type'] = model_type
    eval_data_dict_list.append(eval_data_dict)

    all_real_y_array = eval_data_dict['real'].values.flatten()
    all_pre_y_array = eval_data_dict['predict'].values.flatten()

    filter_index = np.where(abs(all_real_y_array) < 0.1)
    filter_all_real_y_array = np.delete(all_real_y_array, filter_index)
    filter_all_pre_y_array = np.delete(all_pre_y_array, filter_index)
    index_result = model_eval_index(real_y=filter_all_real_y_array, pred_y=filter_all_pre_y_array)
    index_result['type'] = model_type
    index_result_list.append(index_result)

    print('start model GVSAO-LSTM-TCN..........')
    SearchAgents_no = 4
    Max_iter = 2
    lb = [0.001, 1, 32, 3, 1, 32]
    ub = [0.01, 5, 192, 8, 5, 128]
    columns = ['learning_rate', 'num_layers', 'hidden_size', 'kernel_size', 'tcn_layer', 'tcn_hidden_size']

    model_type = 'GVSAO-LSTM-TCN'
    result = GVSAO_optim_model(SearchAgents_no, Max_iter, lb, ub, model_type,
                               train_loader, test_loader_list, epochs, pre_feature_scaler, input_size,
                               predict_date_list, out_param)

    Best_pos = result['Best_pos'].reshape(1, -1)
    Best_para_df = pd.DataFrame(data=Best_pos, columns=columns)
    Best_para_df.to_csv(os.path.join(model_out_dir, 'predict_model_param_best.csv'), index=False)

    Convergence_curve = result['Convergence_curve']

    model = result['model']
    model_filepath = os.path.join(model_out_dir, prefix_name + '-' + model_type + '.pt')
    torch.save(model, model_filepath)

    epoch_list = result['epochs']
    loss_list = result['loss']
    loss_data_list.append({
        'type': model_type,
        'epochs': epoch_list,
        'loss': loss_list,
    })

    eval_data_dict = result['eval']
    eval_data_dict['type'] = model_type
    eval_data_dict_list.append(eval_data_dict)

    index_result = result['index_result']
    index_result['type'] = model_type
    index_result_list.append(index_result)

    #
    loss_df = loss_compare_data(loss_data_list)
    loss_df.to_csv(os.path.join(model_out_dir, 'loss_series_data.csv'))

    #
    predict_result_df = pd.DataFrame()
    for eval_data in eval_data_dict_list:
        predict_df = eval_data['predict']
        real_df = eval_data['real']
        type = eval_data['type']
        predict_df.loc[:, 'type'] = type
        real_df.loc[:, 'type'] = type
        predict_df.loc[:, 'data_type'] = 'predict'
        real_df.loc[:, 'data_type'] = 'real'
        predict_result_df = pd.concat([predict_result_df, predict_df, real_df])
    predict_result_df.to_csv(os.path.join(model_out_dir, 'predict_real_data.csv'))
    #
    fitness_df=fitness_result_out(Max_iter, Convergence_curve)
    fitness_df.to_csv(os.path.join(model_out_dir,  'fitness_series_data.csv'), index=False)
    return


def fitness_result_out(Max_iter, Convergence_curve):
    x = list(range(1, Max_iter + 1))
    fitness_df = pd.DataFrame()
    fitness_df.loc[:, 'iter'] = x
    fitness_df.loc[:, 'fitness'] = Convergence_curve
    return fitness_df


def loss_compare_data(data_model_list):
    loss_df = pd.DataFrame()
    for model_data in data_model_list:
        if loss_df.empty:
            loss_df.loc[:, 'epochs'] = model_data['epochs']
        loss_df.loc[:, model_data['type']] = model_data['loss']
    return loss_df

config = {
    "font.family": 'Times New Roman',
    "axes.unicode_minus": False
}
rcParams.update(config)
plt.rcParams["font.sans-serif"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

if __name__ == '__main__':

    current_data_dir = os.getcwd()
    current_data_dir = os.path.dirname(current_data_dir)

    insar_cluster_dir_name = 'InSAR_Cluster_Result'
    insar_cluster_dir = os.path.join(current_data_dir, insar_cluster_dir_name)

    insar_model_dir_name = 'InSAR_Predict_Result'
    insar_model_dir = os.path.join(current_data_dir, insar_model_dir_name)

    insar_data_filename = r'InSAR_detect_data.csv'
    insar_data_df = pd.read_csv(os.path.join(insar_cluster_dir, insar_data_filename))
    out_param = {}


    type_model_dir = insar_model_dir

    if not os.path.exists(type_model_dir):
        os.makedirs(type_model_dir)

    insar_cluster_filename = r'Cluster_result.csv'
    insar_cluster_df = pd.read_csv(os.path.join(insar_cluster_dir, insar_cluster_filename))
    label_list = list(insar_cluster_df['label'].unique())

    label_list = [12]

    features = ['freq', 'acu_settle']

    for c_label in label_list:
            filter_insar_df = insar_cluster_df[insar_cluster_df['label'] == c_label]

            filter_insar_list = list(filter_insar_df['index'].unique())
            filter_insar_data_df = insar_data_df[insar_data_df['index'].isin(filter_insar_list)]

            out_param['cluster'] = 'cluster' + str(c_label)
            label_type_model_dir = os.path.join(type_model_dir, out_param['cluster'])
            out_param['model_out_dir'] = label_type_model_dir

            if not os.path.exists(label_type_model_dir):
                os.makedirs(label_type_model_dir)
            out_param['prefix_name'] =str(c_label)

            multi_predict_model_main(filter_insar_data_df, insar_list=filter_insar_list, features=features,
                                     out_param=out_param)

            plt.close('all')






