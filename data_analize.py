import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def create_green_table_image(correlation_matrix, velocities, displacements, 
                                     decimal_places=15, filename='high_precision_correlation_table.png'):
    """
    Cria uma imagem de tabela colorida em tons de verde com alta precisão
    
    Args:
        correlation_matrix: matriz de correlações (float64)
        velocities: lista de nomes das velocidades (linhas)
        displacements: lista de nomes dos deslocamentos (colunas)
        decimal_places: número de casas decimais para exibir
        filename: nome do arquivo para salvar a imagem
    """
    # Verificar precisão da matriz
    print(f"\nPrecisão da matriz: {correlation_matrix.dtype}")
    print(f"Valor mínimo: {correlation_matrix.min():.15f}")
    print(f"Valor máximo: {correlation_matrix.max():.15f}")
    
    # Criar figura maior para caber mais texto
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Formatar células com alta precisão
    cell_text = []
    for i in range(correlation_matrix.shape[0]):
        row_text = []
        for j in range(correlation_matrix.shape[1]):
            # Formatar com número específico de casas decimais
            formatted_value = f"{correlation_matrix[i, j]:.{decimal_places}f}"
            row_text.append(formatted_value)
        cell_text.append(row_text)
    
    # Criar tabela
    tabela = ax.table(
        cellText=cell_text,
        rowLabels=velocities,
        colLabels=displacements,
        cellLoc='center',
        loc='center',
        colWidths=[0.18] * len(displacements)  # Colunas mais largas
    )
    
    # Aplicar cores em tons de verde baseado nos valores de correlação
    valores_flat = correlation_matrix.flatten()
    min_val, max_val = valores_flat.min(), valores_flat.max()
    
    # Se todos valores forem iguais, ajustar
    if max_val == min_val:
        max_val = min_val + np.finfo(np.float64).eps
    
    # Aplicar cores às células com alta precisão
    for i in range(len(velocities)):
        for j in range(len(displacements)):
            valor = correlation_matrix[i, j]
            
            # Normalizar com alta precisão
            if max_val > min_val:
                valor_norm = np.float64((valor - min_val) / (max_val - min_val))
            else:
                valor_norm = np.float64(0.5)
            
            # Tons de verde: valor mais alto = verde mais escuro
            # Usar colormap Greens com ajuste para evitar cores muito claras
            cor = plt.cm.Greens(np.float64(0.15 + 0.85 * valor_norm))
            
            # Converter para RGB com precisão
            cor_rgb = (
                float(cor[0]),
                float(cor[1]),
                float(cor[2])
            )
            
            tabela[(i+1, j)].set_facecolor(cor_rgb)
            
            # Melhor contraste para texto usando luminância
            luminancia = np.float64(0.299 * cor_rgb[0] + 0.587 * cor_rgb[1] + 0.114 * cor_rgb[2])
            if luminancia > 0.5:
                tabela[(i+1, j)].set_text_props(color='black', weight='bold', fontsize=9)
            else:
                tabela[(i+1, j)].set_text_props(color='white', weight='bold', fontsize=9)
    
    # Estilizar cabeçalhos com alta precisão nas cores
    for j in range(len(displacements)):
        tabela[(0, j)].set_facecolor('#1B5E20')  # Verde escuro
        tabela[(0, j)].set_text_props(color='white', weight='bold', fontsize=10)
    
    for i in range(len(velocities)):
        tabela[(i+1, -1)].set_facecolor('#2E7D32')  # Verde médio-escuro
        tabela[(i+1, -1)].set_text_props(color='white', weight='bold', fontsize=10)
    
    # Ajustar tamanho da fonte automaticamente
    tabela.auto_set_font_size(False)
    
    # Remover eixos
    ax.axis('off')
    
    # Adicionar título com informações de precisão
    plt.title(f'Matriz de Correlação - Precisão: {decimal_places} casas decimais', 
              pad=25, fontsize=14, fontweight='bold')
    
    # Adicionar barra de cores com precisão
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    norm = Normalize(vmin=min_val, vmax=max_val)
    sm = ScalarMappable(cmap=plt.cm.Greens, norm=norm)
    sm.set_array([])
    
    # Posicionar barra de cores
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', 
                        pad=0.05, fraction=0.03, shrink=0.8)
    cbar.set_label('Valor de Correlação', weight='bold')
    
    # Formatar ticks da barra de cores com alta precisão
    cbar.formatter.set_powerlimits((-3, 3))
    cbar.update_ticks()
    
    # Adicionar informações de precisão
    info_text = f"Precisão: float64 | Min: {min_val:.{decimal_places}f} | Max: {max_val:.{decimal_places}f}"
    plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=9, style='italic')
    
    # Ajustar layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Salvar imagem em alta resolução
    plt.savefig(filename, dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\nTabela com precisão float64 salva como: {filename}")

if __name__ == '__main__':
    df = pd.read_csv('test.csv')
    
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
