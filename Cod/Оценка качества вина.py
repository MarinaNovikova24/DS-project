#!/usr/bin/env python
# coding: utf-8

# Выполните задание по ссылке и оформите в виде CRISP-DM подхода.
# 
# Структурируйте код, отчёт и файлы с данными на основе сегодняшней лекции.
# Загрузите решение в Git и пришлите ссылку на ваш репозиторий.
# 
# Дополнительное задание*
# Попробуйте не загружать CSV-файл с данными, а сделайте отдельный скрипт на его получение.

# # 1. Понимание бизнеса
# 
# ## 1.1 Цель
# Предсказать качество вина
# 
# ## 1.2 Описание
# Информация о наборе данных:
# 
# 
# Эти два набора данных относятся к красному и белому вариантам португальского вина «Винью Верде». Из-за проблем с конфиденциальностью и логистикой доступны только физико-химические (входные) и сенсорные (выходные) переменные (например, нет данных о сортах винограда, марке вина, продажной цене вина и т. д.).
# 
# Эти наборы данных можно рассматривать как задачи классификации или регрессии. Классы упорядочены и не сбалансированы (например, нормальных вин больше, чем отличных или плохих). Алгоритмы обнаружения выбросов могут быть использованы для обнаружения нескольких отличных или плохих вин. Кроме того, мы не уверены, что все входные переменные релевантны. Поэтому было бы интересно протестировать методы выбора признаков.
# 
# Два набора данных были объединены, и несколько значений были случайным образом удалены.
# 
# 
# 

# # 2. Data Understanding
# 
# ## 2.1 Import Libraries

# In[1]:


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import  Normalizer , scale
from sklearn.model_selection import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


# ## 2.2 Вспомогательные функции

# In[2]:


def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_correlation_map( df ):
    corr = data.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )
    
def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    print (model.score( X , y ))
    


# ## 2.3 Загрузка данных

# In[3]:

data.py
#data = pd.read_csv("winequalityN.csv")
#data.shape


# ## 2.4 Статистика и визуализации

# In[4]:


data.head()


# In[5]:


data.quality.unique()


# **Описание переменных**
# 
# 1 - тип вина (белое или красное)
# 
# 2 - фиксированная кислотность
# 
# 3 - летучая кислотность
# 
# 4 - лимонная кислота
# 
# 5 - остаточный сахар
# 
# 6 - хлориды 
# 
# 7 - свободный диоксид серы
# 
# 8 - общий диоксид серы
# 
# 9 - плотность
# 
# 10 - рН
# 
# 11 - сульфаты
# 
# 12 - спирт
# 
# Выходная переменная
# (на основе сенсорных данных):
# 
# 13 - качество (оценка от 0 до 10)

# ### 2.4.1 Далее взглянем на некоторую ключевую информацию о переменных
# Числовая переменная - это переменная со значениями в области целых или действительных чисел, в то время как категориальная переменная - это переменная, которая может принимать одно из ограниченного и обычно фиксированного числа возможных значений, таких как тип вина.
# 

# In[6]:


data.describe()


# In[7]:


data.info()


# In[8]:


describe_more(data)


# ### 2.4.2 Тепловая карта корреляции может дать нам понимание того, какие переменные важны

# In[9]:


plot_correlation_map(data)


# ### 2.4.3 Давайте подробнее рассмотрим взаимосвязь между признаками и качеством вина
# Начнем с рассмотрения взаимосвязи между содержанием спирта и качеством.

# In[10]:


plot_distribution( data , var = 'alcohol' , target = 'quality' , row = 'type' )


# Рассмотрим взаимосвязь между содержанием сахара и качеством.

# In[11]:


plot_distribution( data , var = 'residual sugar' , target = 'quality' , row = 'type' )


# Рассмотрим графики выше. Различия между выживаемостью для разных значений - это то, что будет использоваться для разделения целевой переменной (в данном случае - выживаемости) в модели. Если бы две линии были примерно одинаковыми, то это не было бы хорошей переменной для нашей прогностической модели.

# ### 2.4.4 Тип вина
# Мы также можем посмотреть на категориальную переменную - тип вина и ее связь с оценкой

# In[12]:


# Plot quality rate by type
plot_categories( data , cat = 'type' , target = 'quality' )


# # 3. Data Preparation

# ## 3.1 Категориальные переменные должны быть преобразованы в числовые переменные
# 
# Переменные Embarked, Pclass и Sex рассматриваются как категориальные переменные. Некоторые из  алгоритмов могут обрабатывать только числовые значения, поэтому нам нужно создать новую (фиктивную) переменную для каждого уникального значения категориальных переменных (OneHotEncoding)

# In[13]:


# Transform type into binary values 0 and 1
color = pd.Series( np.where( data['type'] == 'red' , 1 , 0 ) , name = 'color' )


# ## 3.2 Заполнить пропущенные значения в переменных
# Большинство алгоритмов машинного обучения требуют, чтобы все переменные имели значения, чтобы использовать их для обучения модели. Самый простой метод - заполнить пропущенные значения средним по переменной для всех наблюдений в обучающем наборе.

# In[14]:


df = pd.DataFrame()
for i in data.loc[:, data.columns !='type']:
  df[i] = data[i].groupby(data['type']).fillna(data[i].mean())


# In[15]:


df = df.drop(['total sulfur dioxide', 'quality'], axis=1)


# In[16]:


df.info()


# ## 3.3 Feature Engineering &ndash; добавляем новые признаки
# 

# **На основе признака - содержания сахара - создадим новые признаки:**
# 
# Сухие натуральные вина (до 5 г/л).
# 
# Полусухие вина (от 5 до 30 г/л).
# 
# Полусладкие вина (от 30 до 80 г/л).
# 
# Сладкое (от 80 г/л).
# 
# 
# 
# 

# In[17]:


def sugar(x):
  if x < 5:
    return 'dry'
  elif x >= 5 and x < 30:
    return 'semi-dry'
  elif x >= 30 and x < 80:
    return 'semi-sweet'
  else:
    return 'sweet'   


# In[18]:


title = pd.DataFrame()
# we extract the title from each name
title[ 'Title' ] = data['residual sugar'].apply(sugar)


# In[19]:


title


# In[20]:


title = pd.get_dummies( title.Title )
title.head()


# In[21]:


data.quality.value_counts()


# ## 3.4 Сборка финальных датасетов для моделирования

# ### 3.4.1 Variable selection
# Выбираем признаки для формирования итогового датасет.
# 
# 

# In[22]:


full_X = pd.concat( [ df , title , color ] , axis=1 )
full_X.head()


# ### 3.4.2 Создание датасетов
# 
# Отделяем данные для обучения и для проверки

# In[23]:


# Create all datasets that are necessary to train, validate and test models
train_valid_X = full_X[ 0:6497 ]
train_valid_y = data.quality
test_X = full_X[:]
train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )

print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)


# ### 3.4.3 Важность признаков
# Отбор оптимальных признаков для модели имеет важное значение. Теперь мы попытаемся оценить, какие переменные являются наиболее важными, чтобы сделать прогноз.

# In[24]:


plot_variable_importance(train_X, train_y)


# # 4. Моделирование
# Теперь мы выберем модель, которую хотели бы попробовать. Используем обучающий набор данных для обучения модели и затем проверим ее с помощью тестового набора.
# 
# ## 4.1 Выбор модели
# Хорошей отправной точкой является логистическая регрессия.

# In[25]:


model = RandomForestClassifier(max_depth = 13, random_state = 13)


# ## 4.2 Обучение модели

# In[26]:


model.fit( train_X , train_y )


# # 5. Оценка
# Теперь мы собираемся оценить модель
# 
# ## 5.1 Модель
# Мы можем оценить точность модели, используя набор для валидации, где мы знаем фактический результат. Этот набор данных не использовался для обучения, поэтому он абсолютно новый для модели.
# 
# Затем мы сравниваем точность с точностью при использовании модели на тренировочных данных. Если разница между ними значительна, это свидетельствует о переобучении. Мы стараемся избегать этого, потому что это означает, что модель не будет хорошо обобщаться на новые данные (будет работать плохо)

# In[27]:


print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))


# # 6. Развертывание
# 
# Развертывание в данном означает публикацию полученного прогноза в таблицу лидеров Kaggle.

# In[28]:


test_Y = model.predict( test_X )
id = data[:6497].index
test = pd.DataFrame( { 'id': id , 'quality': test_Y } )
test.shape
test.head()
test.to_csv( 'wine_pred.csv' , index = False )

