import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy
from utils import get_viewport_tiles
from collections import Counter

def get_correlation(df, col_1, col_2):
    corr = df[col_1].corr(df[col_2], 'pearson')
    return -10 * np.log10(1 - corr)

def get_correlation_all(df, velocities, displacements):

    correlation_matrix = []
    for v in velocities:
        velocity_correlation = []
        for d in displacements:
            corr = get_correlation(df, v, d)
            print(f'correlation {v} to {d}: {corr}')
            velocity_correlation.append(corr)
        
        correlation_matrix.append(velocity_correlation)

    return np.array(correlation_matrix)

def calcular_entropia(data):
    val, count = np.unique(data, return_counts=True)

    prob = count/len(data)
    return entropy(prob, base=2)

def calcular_entropia_conjunta(pitch_series, yaw_series, bins=None):
    if bins is None:
        # Regra de Sturges
        n = len(pitch_series)
        bins = int(np.ceil(np.log2(n)) + 1) if n > 0 else 10

    pitch_discreto = pd.qcut(pitch_series, q=bins, labels=False, duplicates='drop')
    yaw_discreto = pd.qcut(yaw_series, q=bins, labels=False, duplicates='drop')

    pares = list(zip(pitch_discreto, yaw_discreto))

    contador = Counter(pares)
    total_pares = len(pares)

    entropia_conjunta = 0.0
    for count in contador.values():
        if count > 0:
            p = count / total_pares
            entropia_conjunta -= p * np.log2(p)

    return entropia_conjunta

def entropy_all(df):
    resultados = []

    for (v_id, u_id), df_group in df.groupby(['v_id', 'u_id']):
        ent_conjunta = calcular_entropia_conjunta(df_group['pitch'], df_group['yaw'])

        resultados.append({
            'v_id': v_id,
            'u_id': u_id,
            'entropia_conjunta': ent_conjunta
        })

        print(f"Video {v_id}, Usuário {u_id}: "
              f"joint_entropy={ent_conjunta:.4f}}")

    return pd.DataFrame(resultados)

def get_hmd_traces(df):
    
    for(v_id, u_id), df_s in df.groupby(['u_id', 'v_id']):
        with open(f"tracesHMD/user{v_id}_{u_id}.csv", 'x') as f:
            header = ','.join([f'"V{i}"' for i in range(1, 40)])
            f.write(f"{header}\n")
            last_tiles = []

            for index, row in df_s.iterrows():
                tiles = get_viewport_tiles(row['pitch'], row['yaw'])
                if tiles != last_tiles:
                    line = f"{row['playback_time']}"

                    for tile in tiles:
                        line += f",{tile}"

                    current_columns = 1 + len(tiles)

                    line += ",NA" * (40 - current_columns)

                    f.write(f"{line}\n")
                    last_tiles = tiles

if __name__ == '__main__':
    df = pd.read_csv('test_data.csv')
    df = df.dropna()
    entropy_df = entropy_all(df)
    entropy_df.to_csv('entropy.csv', index=False)
    
    #df = pd.read_csv('test_data.csv')
    #get_hmd_traces(df)
    
    """
    pitch_velocities = ['pitch_v', 'pitch_ema_1', 'pitch_ema_2', 'pitch_ema_3']
    pitch_displacements = ['pitch_d_1', 'pitch_d_2', 'pitch_d_3', 'pitch_d_4', 'pitch_d_5']

    pitch_corr = get_correlation_all(df, pitch_velocities, pitch_displacements)
    print(pitch_corr)

    create_green_table_image(
            correlation_matrix=pitch_corr,
            velocities=pitch_velocities,
            displacements=pitch_displacements,
            filename='correlacao_pitch.png'
        )
    """
    #new_df = df[['playback_time', 'pitch', 'yaw', 'pitch_v', 'yaw_v', 'pitch_pred_3', 'yaw_pred_3']].copy()
    #print(new_df.head())
    #print(df.info())

