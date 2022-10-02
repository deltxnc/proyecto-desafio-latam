import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import missingno as msno

warnings.filterwarnings(action= 'ignore')
plt.style.use('seaborn-whitegrid')

def list_na(dataframe, var, print_list = False):
    """Retorna la cantidad de casos perdidos y el porcentaje correspondiente

    :dataframe: La función debe ingresar un objeto DataFrame.
    :type dataframe: dataframe
    :var: Variable a inspeccionar.
    :type: string
    :print_list: Opción para imprimir la lista de observaciones perdidas en la variable. 
    :type: bool
    :return: cantidad, porcentaje, dataframe
    :rtype: int, float, dataframe
    """    
    print ("####################",var,":", dataframe[var].dtype,"######################")
    print("La cantidad de registros perdidos es:", dataframe[var].isnull().sum())
    print("El porcentaje de registros perdidos es:", (dataframe[var].isnull().sum() / len(dataframe[var]) * 100))
    
    if print_list:
        return dataframe[dataframe[var].isnull()]


def grafestad_numbers(dataframe, ajuste):
    '''
    Devuelve el histograma de las frecuencias de atributos de tipo float o int. Adicionalmente, genera dos líneas verticales que indican la media (rojo) y moda (azul) del atributo analizado.
    ----------
    Elementos:
        
    - dataframe: Dataframe a analizar
    - ajuste: (int) valor de separación lateral de los gráficos.
        
    '''
    variable = []
    for col in dataframe:
        if (dataframe[col].dtypes == 'float64') or (dataframe[col].dtypes == 'int64') or (dataframe[col].dtypes == 'int32'):
            variable.append(col)
    for i in range (len(variable)):
        plt.subplot(2,len(variable),i+1)
        plt.hist(dataframe[variable[i]].dropna())
        plt.title(f'Frecuencias de {variable[i]}')
        plt.axvline(dataframe[variable[i]].mean(), color= 'tomato')
        plt.axvline(dataframe[variable[i]].median(), color= 'blue')
        plt.xlabel('')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=30)
    return plt.subplots_adjust(right=ajuste, hspace=.5)

def grafestad_object(dataframe, ajuste):
    '''
    Devuelve un gráfico de barras de las frecuencias de atributos de tipo object.
    Elementos:
        
    - dataframe: Dataframe a analizar
    - ajuste: (int) valor de separación lateral de los gráficos.
        
    '''
    variable = []
    for col in dataframe:
        if dataframe[col].dtypes == 'object':
            variable.append(col)
    for i in range (len(variable)):
        plt.subplot(2,len(variable),i+1)
        sns.countplot(dataframe[variable[i]].dropna())
        plt.title(f'Frecuencias de {variable[i]}')
        plt.xlabel('')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=90)
    return plt.subplots_adjust(right=ajuste, hspace=.5)

def box_plot(df, name):
    '''
    Devuelve un grafico box plot.
    Elementos:
        -df: dataframe a analizar
        -name: nombre de atributo a graficar
    '''
    sns.boxplot(x="TARGET", y=name, data=df)
    plt.title(f"Box plot {name}")
    plt.show()

def densidad(df, var, log = False):
    '''
      Grafica un histograma señalando las curvas de densidad.
      Elementos:
          -df: La función debe ingresar un objeto DataFrame.
          -var: Variable a inspeccionar.
          -log: metodo de normalizacion, False por defecto para estudiar la variable sin normalizacion
    '''
    df_var = df[var]
    
    np.mean(df_var)
    log_df = np.log(df_var)

    plt.figure(figsize=(15,5))
    if log == False:
        plt.subplot(1, 2, 1)
        plt.hist(df_var, color='grey', alpha=.4, density=True)

        x_min, x_max = plt.xlim()
        x_axis = np.linspace(x_min, x_max, 100)

        plt.plot(x_axis, stats.norm.pdf(x_axis,df_var.mean(),df_var.std()),color='tomato', lw=3)
        plt.axvline(df_var.mean(),color='dodgerblue',linestyle='--', lw=3, label='Promedio')
        plt.title(f"Curva de Densidad de {var}.")

    if log == True:
        plt.subplot(1, 2, 2)
        plt.hist(log_df, color='grey', alpha=.4, density=True)
        x_min, x_max = plt.xlim()
        x_axis = np.linspace(x_min, x_max, 100)

        plt.plot(x_axis, stats.norm.pdf(x_axis,log_df.mean(),log_df.std()),color='tomato', lw=3, label=var)
        plt.axvline(log_df.mean(),color='dodgerblue',linestyle='--', lw=3, label='Promedio')
        plt.title(f"Curva log_df {var}.")
    plt.legend()

def plt_density(df, var):
    """Grafica un histograma señalando las curvas de densidad

    :df: La función debe ingresar un objeto DataFrame.
    :type dataframe: dataframe
    :var: Variable a inspeccionar.
    :type: string
    """

    df_dropna = df[var].dropna()

    # Con plt.subplot vamos a dividir el espacio del en dos partes
    #plt.subplot(2,1,1)
    # Graficamos el mismo histograma, especificando density para que el histograma represente densidades y no frecuencias
    plt.hist(df_dropna, color='grey', alpha=.4, density=True)
    # extraemos los límites del histograma
    x_min, x_max = plt.xlim()
    # utilizandos los límites del histograma para crear un array
    x_axis = np.linspace(x_min, x_max, 100)
    # graficamos la curva de densidad empirica (permite comparar directamente con la curva de densidad teorica)
    gauss_kde = stats.gaussian_kde(df_dropna) # Kernel gaussiano
    Z = np.reshape(gauss_kde(x_axis).T, x_axis.shape)
    plt.plot(x_axis, Z, color='tomato', lw=3)
    # agregamos la línea vertical para identificar la media
    plt.axvline(df_dropna.mean(), color='dodgerblue', linestyle='--', lw=3, label='media')
    plt.title("Histograma de densidad para " +  var)
    plt.legend()
    