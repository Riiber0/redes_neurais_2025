import numpy as np
import pandas as pd
import os, sys
import progressbar

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ['TF_NUM_INTEROP_THREADS'] = '2' 
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import R2Score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import deque
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from utils import prepare_data, split_dataset, create_temporal_sequences, create_detailed_temporal_sequences,get_viewport_tiles, sin_convert
from model import CNN_GRU, baseModel, CNN_LSTM
import glob

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

MODELS_PATH = './models/'
TILE_RESULT_PATH = './tile_result/'

def create_model(model_type, name, x_train, y_train, x_val, y_val):
    if model_type == 'CNN':
        m = baseModel(name)
    elif model_type == 'CNN_GRU':
        m = CNN_GRU(name)
    elif model_type == 'CNN_LSTM':
        m = CNN_GRU(name)

    m.create_regression_model(x_train.shape[1:])
    m.compile_model()
    m.model_summary()

    m.train_model(x_train, y_train, x_val, y_val, b_size=512)


def make_data(df, p_target, y_target):
    scaled_df = prepare_data(df, p_target, y_target)

    return scaled_df

def get_aer(model_path, x_test, y_target):
    model_name = model_path.split('/')[2]
    model_name = model_name.split('.')[0]
    print(model_name)

    model = load_model(model_path)

    preds = model.predict(x_test)
    mse = mean_squared_error(y_target, preds)
    print(f"MSE: {mse}")

def get_mae(pred, target):
    erro = np.abs(target - pred)
    n = len(pred)

    mae = np.mean(erro)
    se = stats.sem(erro)

    mae_lower, mae_upper = stats.interval(0.95, n - 1, loc = mae, scale = se)

    return mae, mae_lower, mae_upper

def get_rmse(pred, target):
    erro = np.abs(target - pred)
    n = len(pred)

    mse = np.mean(erro**2)
    rmse = np.mean(erro)

    alpha = 1 - 0.95
    rmse_lower = stats.chi2.ppf(alpha/2, (n-1))
    rmse_upper = stats.chi2.ppf(1-alpha/2, (n-1))

    return rmse, rmse_lower, rmse_upper

def model_stat(model_path, x_test, y_target):
    model_name = model_path.split('/')[2]
    model_name = model_name.split('.')[0]
    print(model_name)

    model = load_model(model_path)

    preds = model.predict(x_test)

    n = len(preds)
    error = np.abs(y_target - preds)

    result = dict()
    result['mae'], se = np.mean(error), stats.sem(error)
    result['mae_lower'], result['mae_upper'] = stats.t.interval(0.95, n - 1, loc=result['mae'], scale=se)

    mse = np.mean(error**2)
    rmse = np.sqrt(mse)
    result['rmse'] = rmse

    alpha = 1 - 0.95
    chi2_lower = stats.chi2.ppf(alpha / 2, (n-1))
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, (n-1))

    result['rmse_lower'] = np.sqrt(((n-1) * rmse**2) / chi2_upper)
    result['rmse_upper'] = np.sqrt(((n-1) * rmse**2) / chi2_lower)

    return result

def get_stats(train_df):

    models = glob.glob(MODELS_PATH+'*.h5')
    result_final = []
    print(models)

    for model in models:
        if 'CNN_GRU' in model or 'CNN_LSTM' in model:
            target_id = model.split('_')[2]
            p_target = 'pitch_pred_' + target_id
            y_target = 'yaw_pred_' + target_id
            target_id = int(target_id)
            pred_time = 5 * target_id
            use_v = True
        else:
            target_id = model.split('_')[1]
            p_target = 'pitch_pred_' + target_id
            y_target = 'yaw_pred_' + target_id
            target_id = int(target_id)
            pred_time = 5 * target_id
            use_v = False

        model_name = model.split('/')[2]
        model_name = model_name.split('.')[0]

        dfs = []
        dfs.append(train_df[train_df['v_id'].between(0, 6)])
        dfs.append(train_df[train_df['v_id'].between(7, 16)])
        dfs.append(train_df[train_df['v_id'].between(33, 41)])
        dfs.append(train_df[train_df['v_id'].between(42, 59)])
        dfs.append(train_df[train_df['v_id'].between(60, 87)])
        i = 1
        for df in dfs:
            print(f"{model_name} - {i}:")
            test_data = make_data(df, p_target, y_target)
            x_test, y_test = create_temporal_sequences(test_data, pred_time, use_v)

            r = model_stat(model, x_test, y_test)
            r['model'] = model_name
            r['pred_time'] = pred_time
            r['dataset.work'] = 'dataset ' + str(i)

            result_final.append(r)
            i += 1

        
        print(f"geral:\n")
        test_data = make_data(train_df, p_target, y_target)
        x_test, y_test = create_temporal_sequences(test_data, pred_time, use_v)
        r = model_stat(model, x_test, y_test)
        r['model'] = model_name
        r['pred_time'] = pred_time
        r['dataset.work'] = 'Geral' 
        result_final.append(r)

    df = pd.DataFrame(result_final)
    df.to_csv('erro_per_dataset.csv')
    #get_aer(model, x_test, y_target)

def create_all(train_df, val_df):
    exp = []
    exp_1 = {'pred_time': 5,
            'p_target': 'pitch_pred_1',
            'y_target': 'yaw_pred_1',
            'target_id': 1
            }
    exp.append(exp_1)

    exp_2 = {'pred_time': 10,
            'p_target': 'pitch_pred_2',
            'y_target': 'yaw_pred_2',
            'target_id': 2
            }
    exp.append(exp_2)

    exp_3 = {'pred_time': 15,
            'p_target': 'pitch_pred_3',
            'y_target': 'yaw_pred_3',
            'target_id': 3
            }
    exp.append(exp_3)

    exp_4 = {'pred_time': 20,
            'p_target': 'pitch_pred_4',
            'y_target': 'yaw_pred_4',
            'target_id': 4
            }
    exp.append(exp_4)

    exp_5 = {'pred_time': 25,
            'p_target': 'pitch_pred_5',
            'y_target': 'yaw_pred_5',
            'target_id': 5
            }
    exp.append(exp_5)

    for i in exp:
        train_data = make_data(train_df, i['p_target'], i['y_target'])
        val_data = make_data(val_df, i['p_target'], i['y_target'])

        x_train, y_train = create_temporal_sequences(train_data, i['pred_time'], True)
        x_val, y_val = create_temporal_sequences(val_data, i['pred_time'], True)
        create_model('CNN_GRU', f"CNN_GRU_{i['target_id']}", x_train, y_train, x_val, y_val)

        x_train, y_train = create_temporal_sequences(train_data, i['pred_time'], False)
        x_val, y_val = create_temporal_sequences(val_data, i['pred_time'], False)
        create_model('CNN', f"CNN_{i['target_id']}", x_train, y_train, x_val, y_val)

        """
        x_train, y_train = create_temporal_sequences(train_data, i['pred_time'], True)
        x_val, y_val = create_temporal_sequences(val_data, i['pred_time'], True)
        create_model('CNN_LSTM', f"CNN_LSTM_{i['target_id']}", x_train, y_train, x_val, y_val)
        """

"""
def get_pitch_yaw(model_path, x_test):
    model_name = model_path.split('/')[2]
    model_name = model_name.split('.')[0]
    print(model_name)

    model = load_model(model_path)

    preds = model.predict(x_test)

    pred_proc = []
    for p in preds:
        pitch, yaw = sin_convert(p[0], p[2])
        data = {'pitch': pitch,
                'yaw': yaw
                }
        pred_proc.append(data)

    pd.DataFrame(pred_proc).to_csv('angles/'+model_name+'.csv', index=False)
"""

def get_pitch_yaw(data):
    pitch_ret = []
    yaw_ret = []
    for p in data:
        pitch, yaw = sin_convert(p[0], p[2])
        pitch_ret.append(pitch)
        yaw_ret.append(yaw)

    pitch_ret = np.array(pitch_ret)
    yaw_ret = np.array(yaw_ret)
    return pitch_ret, yaw_ret

def get_pitch_yaw_all(train_df):

    models = glob.glob(MODELS_PATH+'*.h5')
    print(models)

    for model in models:
        if 'CNN_GRU' in model or 'CNN_LSTM' in model:
            target_id = model.split('_')[2]
            p_target = 'pitch_pred_' + target_id
            y_target = 'yaw_pred_' + target_id
            target_id = int(target_id)
            pred_time = 5 * target_id
            use_v = True
        else:
            target_id = model.split('_')[1]
            p_target = 'pitch_pred_' + target_id
            y_target = 'yaw_pred_' + target_id
            target_id = int(target_id)
            pred_time = 5 * target_id
            use_v = False

        test_data = make_data(train_df, p_target, y_target)

        x_test, y_target = create_temporal_sequences(test_data, pred_time, use_v)
        get_pitch_yaw(model, x_test)

def create_all(train_df, val_df):
    exp = []
    exp_1 = {'pred_time': 5,
            'p_target': 'pitch_pred_1',
            'y_target': 'yaw_pred_1',
            'target_id': 1
            }
    exp.append(exp_1)

    exp_2 = {'pred_time': 10,
            'p_target': 'pitch_pred_2',
            'y_target': 'yaw_pred_2',
            'target_id': 2
            }
    exp.append(exp_2)

    exp_3 = {'pred_time': 15,
            'p_target': 'pitch_pred_3',
            'y_target': 'yaw_pred_3',
            'target_id': 3
            }
    exp.append(exp_3)

    exp_4 = {'pred_time': 20,
            'p_target': 'pitch_pred_4',
            'y_target': 'yaw_pred_4',
            'target_id': 4
            }
    exp.append(exp_4)

    exp_5 = {'pred_time': 25,
            'p_target': 'pitch_pred_5',
            'y_target': 'yaw_pred_5',
            'target_id': 5
            }
    exp.append(exp_5)

    for i in exp:
        train_data = make_data(train_df, i['p_target'], i['y_target'])
        val_data = make_data(val_df, i['p_target'], i['y_target'])

        """
        x_train, y_train = create_temporal_sequences(train_data, i['pred_time'], True)
        x_val, y_val = create_temporal_sequences(val_data, i['pred_time'], True)
        create_model('CNN_GRU', f"CNN_GRU_{i['target_id']}", x_train, y_train, x_val, y_val)

        x_train, y_train = create_temporal_sequences(train_data, i['pred_time'], False)
        x_val, y_val = create_temporal_sequences(val_data, i['pred_time'], False)
        create_model('CNN', f"CNN_{i['target_id']}", x_train, y_train, x_val, y_val)
        """
        x_train, y_train = create_temporal_sequences(train_data, i['pred_time'], True)
        x_val, y_val = create_temporal_sequences(val_data, i['pred_time'], True)
        create_model('CNN_LSTM', f"CNN_LSTM_{i['target_id']}", x_train, y_train, x_val, y_val)

def test_model(model_path, x_test, y_test):
    m = load_model(model_path)
    test_results = m.evaluate(x_test, y_test)

    print(m.name)
    print(f"Loss: {test_results[0]}")
    print(f"MAE: {test_results[1]}")
    print(f"RMSE: {test_results[2]}")
    print(test_results)

    return [test_results[0], test_results[1], test_results[2]]

def test_all(test_df):

    models = glob.glob(MODELS_PATH+'*.h5')
    print(models)
    result_final = []

    for model in models:
        if 'CNN_GRU' in model or 'CNN_LSTM' in model:
            target_id = model.split('_')[2]
            p_target = 'pitch_pred_' + target_id
            y_target = 'yaw_pred_' + target_id
            target_id = int(target_id)
            pred_time = 5 * target_id
            use_v = True
        else:
            target_id = model.split('_')[1]
            p_target = 'pitch_pred_' + target_id
            y_target = 'yaw_pred_' + target_id
            target_id = int(target_id)
            pred_time = 5 * target_id
            use_v = False

        dfs = []
        dfs.append(test_df[test_df['v_id'].between(0, 6)])
        dfs.append(test_df[test_df['v_id'].between(7, 16)])
        dfs.append(test_df[test_df['v_id'].between(33, 41)])
        dfs.append(test_df[test_df['v_id'].between(42, 59)])
        dfs.append(test_df[test_df['v_id'].between(60, 87)])
        i = 1
        for df in dfs:
            print(f"{model} - {i}:\n")
            test_data = make_data(df, p_target, y_target)

            x_test, y_test = create_temporal_sequences(test_data, pred_time, use_v)
            r = test_model(model, x_test, y_test)
            result_final.append({ 'model': model,
                                'pred_time': pred_time,
                                'dataset.work': 'dataset '+str(i),
                                'loss' : r[0],
                                'mae' : r[1],
                                'rmse' : r[2]
                })
            i += 1

    df = pd.DataFrame(result_final)
    df.to_csv('erro_per_dataset.csv')


def test_tile_miss(model_path, test_data, x_test, pred_time):
    model_name = None
    if 'CNN_GRU' in model_path:
        model_name = 'CNN_GRU'
    elif 'CNN_LSTM' in model_path:
        model_name = 'CNN_LSTM'
    elif 'CNN' in model_path:
        model_name = 'CNN'

    results = []

    m = load_model(model_path)
    predictions = m.predict(x_test)

    entropy_df = pd.read_csv('entropy_classified.csv')

    end = len(test_data)
    cur = 0
    widgets = [
    'Progress: ',                               
    progressbar.Counter(format='%(value)d/%(max_value)d'),
    ' ',
    progressbar.Bar(marker=':'),      
    ' ',
    progressbar.ETA(),                    
    ' | ',
    progressbar.Timer(),                  
    ]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=end, end='\n')

    pred_index = 0
    model_sum = deque()
    for (v_id, u_id), session in test_data.groupby(['v_id', 'u_id']):
        video_buffer = {}
        video_buffer[1] = [i for i in range(1,201)]

        segment_tiles = {}
        segment_tiles[1] = []

        pred_tiles = []
        total_tile_miss = 0
        session_sum = {}

        for i in range(2, 16):
            session_sum['tile_miss_'+str(i)] = 0
            video_buffer[i] = []
            segment_tiles[i] = []

        trace_class = entropy_df[(entropy_df['u_id'] == u_id) & (entropy_df['v_id'] == v_id)]['classificacao']
        #print(f"v_id:{v_id}, u_id:{u_id}")
        #print(trace_class.iloc[0])
        trace_class = trace_class.iloc[0]

        for index, row in session.iterrows():
            tile_miss = 0
            download_size = 0
            fov_tiles = 0
            wasted_tiles = 0
            if row['sequence']:
                #pred = m.predict(np.array([row['sequence']]), batch_size=1, verbose=0)[0]
                #print(f"{pred} - {predictions[pred_index]}")
                pred = predictions[pred_index]
                pred_tiles = get_viewport_tiles(pred[0], pred[1], pred[2], pred[3])
                pred_index+=1

            current_seg = int(row['playback_time']//4 + 1)
            pred_seg = int((row['playback_time']+pred_time)//4 + 1)
            if pred_seg > 15:
                break

            tiles_to_download = [tile for tile in pred_tiles if tile not in video_buffer[pred_seg]]
            download_size = len(tiles_to_download)
            video_buffer[pred_seg] += tiles_to_download
            video_buffer[pred_seg].sort()

            #print(f"{index}: current segment = {current_seg}, playback time = {row['playback_time']}")
            #print(get_viewport_tiles(row['target'][0], row['target'][2]))
            #print(video_buffer[current_seg])

            view_port_tiles = get_viewport_tiles(row['target'][0], row['target'][1], 
                    row['target'][2], row['target'][3])

            fov_tiles = len(view_port_tiles)
            segment_tiles[current_seg] += set(view_port_tiles) - set(segment_tiles)
            wasted_tiles = len(set(video_buffer[current_seg]) - set(segment_tiles[current_seg]))
            """
            if row['playback_time'] == 4.0:
                print('====================')
                print(current_seg)
                print(segment_tiles[current_seg])
                print(video_buffer[current_seg])
                print(set(video_buffer[current_seg]) - set(segment_tiles[current_seg]))
                sys.exit()
            """

            if not set(view_port_tiles).issubset(set(video_buffer[current_seg])):
                #print(set(view_port_tiles) - set(video_buffer[current_seg]))
                #print(view_port_tiles)
                #print(video_buffer[current_seg])

                missed_tiles = set(view_port_tiles) - set(video_buffer[current_seg])
                total_tile_miss += len(missed_tiles)
                session_sum['tile_miss_'+str(current_seg)] += len(missed_tiles)

                tile_miss = len(missed_tiles)

                video_buffer[current_seg] += list(missed_tiles)

            line_result = {'u_id': u_id,
                            'v_id': v_id,
                            'trace_class': trace_class,
                            'playback_time': row['playback_time'],
                            'segment': current_seg,
                            'download_size': download_size,
                            'tile_miss': tile_miss,
                            'fov_tiles': fov_tiles,
                            'wasted_tiles': wasted_tiles
                    }

            model_sum.append(line_result)

            cur += 1
            bar.update(cur)
        
        session_sum['u_id'] = u_id
        session_sum['v_id'] = v_id
        session_sum['total_tile_miss'] = total_tile_miss
        session_sum['model'] = model_name
        session_sum['pred_time'] = pred_time
        session_sum['trace_class'] = trace_class
                        
        results.append(session_sum)
        model_df = pd.DataFrame(model_sum)
        model_df.to_csv(TILE_RESULT_PATH+model_name+str(int(pred_time*10))+'.csv')

    bar.finish()
    return results
        


def test_tiles_all(test_df):

    models = glob.glob(MODELS_PATH+'*.h5')
    print(models)
    final_results = []

    for model in models:
        if 'CNN_GRU' in model or 'CNN_LSTM' in model:
            target_id = model.split('_')[2]
            p_target = 'pitch_pred_' + target_id
            y_target = 'yaw_pred_' + target_id
            target_id = int(target_id)
            pred_time = 5 * target_id
            use_v = True
        else:
            target_id = model.split('_')[1]
            p_target = 'pitch_pred_' + target_id
            y_target = 'yaw_pred_' + target_id
            target_id = int(target_id)
            pred_time = 5 * target_id
            use_v = False

        test_data = make_data(test_df, p_target, y_target)

        x_test, y_test = create_temporal_sequences(test_data, pred_time, use_v)
        test_data = create_detailed_temporal_sequences(test_data, pred_time, use_v)
        #model_name = model.split('.')[1]
        #model_name = model_name.split('/')[2]
        #print(model_name)
        results = test_tile_miss(model, test_data, x_test, pred_time/10)
        final_results += results
        #ite_df = pd.DataFrame(results)
        #ite_df.to_csv(f"{model_name}_tiles.csv")

    final_df = pd.DataFrame(final_results)
    final_df.to_csv('tiles_results.csv')

if __name__ == '__main__':
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
                
    # Limitar uso de memória
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Aviso GPU: {e}")

    np.random.seed(9000)

    if len(sys.argv) < 2:
        print('help:\n\tcreate: create models and data\n\ttest: test models in models directory')
        sys.exit(0)

    elif sys.argv[1] == 'create':
        raw_data = pd.read_csv('dataset_processed.csv')
        raw_data = raw_data.dropna()

        train_df, val_df, test_df = split_dataset(raw_data)
        test_df.to_csv('test_data.csv')

        create_all(train_df, val_df)

    elif sys.argv[1] == 'test':
        test_df = pd.read_csv('test_data.csv')
        test_all(test_df)

    elif sys.argv[1] == 'tiles':
        #print(get_viewport_tiles(0,0))
        test_df = pd.read_csv('test_data.csv')
        #for (u_id, v_id), goup in test_df.groupby(['u_id', 'v_id']):
        #    print(f"Video {v_id}, Usuário {u_id}")

        test_tiles_all(test_df)
    elif sys.argv[1] == 'stat':
        test_df = pd.read_csv('test_data.csv')
        get_stats(test_df)

    elif sys.argv[1] == 'angles':
        test_df = pd.read_csv('test_data.csv')
        get_pitch_yaw_all(test_df)

    else:
        m = load_model('./models/CNN_4_model.h5')
        test_df = pd.read_csv('test_data.csv')
        test_data = make_data(test_df, 'pitch_pred_4', 'yaw_pred_4')

        x_test, y_test = create_temporal_sequences(test_data, 20, False)
        temp_d = create_detailed_temporal_sequences(test_data, 20, False)
        print(f"{x_test[0]}")
        print(f"{temp_d['sequence'].iloc[19]}")
        print((x_test[0] == temp_d['sequence'].iloc[19]))
        if x_test[0].all(temp_d['sequence'].iloc[19]):
            print('ingual')





