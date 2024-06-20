# import os
# import sys

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# # from sklearn.preprocessing import OneHotEncoder, StandardScaler
# # from sklearn.compose import ColumnTransformer
# # from sklearn.pipeline import Pipeline
# # from catboost import CatBoostRegressor
# import streamlit as st
# from PIL import Image
# # from my_module import LotFrontageImputer
# from category_encoders import OrdinalEncoder
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# import pickle

# from shapash import SmartExplainer
# from shapash.data.data_loader import data_loading
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from sklearn.impute import SimpleImputer 

# train = pd.read_csv("./house-prices-advanced-regression-techniques/train.csv")


# Browse model !!!
def load_model():
    with open("house_pricing_pipeline.pkl", "rb") as file:
        model = pickle.load(file)
    return model


# Функция для загрузки данных и предсказания
def predict(data):
    model = load_model()
    prediction = model.predict(data)
    return prediction

# Функция для вычисления метрик
def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


# Загрузка данных
def load_data(file):
    data = pd.read_csv(file)
    return data



# Функция для EDA
def exploratory_data_analysis(data):
    st.write(data.head())




# Главная часть приложения
    
def show_main_info():
    st.markdown("""
                <h1 style='color: #FE5F55; text-align: center; 
                padding: 20px; font-size: 36px; background-color: #F0F6F7FF;'>
                🏠📈 Модель прогнозирования цен на недвижимость 📉🏠
                </h1>
                """, unsafe_allow_html=True)
    
    background_image_path = "https://media.istockphoto.com/id/1288957751/vector/traditional-european-style-houses-in-old-town-neighborhood-suburban-traditional-street.jpg?s=612x612&w=0&k=20&c=ldiv_9BlStJdCK95gDCjvG7cwZzeuRmUBHMsZBvl26Y="
    st.image(background_image_path, caption='', use_column_width=True)


    

    st.write("## Этапы проекта")

    st.write("В этом проекте мы использовали данные из соревнования на Kaggle [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview).")
    
    
    st.write("### 1. Изучение данных")
    st.write("- Заполнили пропущенные значения по среднему значению, по моде, некоторые столбцы - нулями;")
    st.write("- Удалили фичи, которые не обладают статистической значимостью;")
    st.write("- Проверили на наличие выбросов;")
    st.write("- Нормализация данных с помощью StandardScaler, OneHotEncoder и OrdinalEncoder")

    st.write("### 2. Обучение модели")
    st.write("Мы использовали следующие модели для обучения:")
    st.write("- LinearRegression")
    st.write("- GradientBoostingRegressor")
    st.write("- CatBoostRegressor")
    st.write("- RandomForestRegressor")
    st.write("- LGBMRegressor")
    st.write("- XGBRegressor")
    st.write("- VotingRegressor")

    st.write("### 5. Результаты")
    st.write("Лучший результат был достигнут при использовании модели CatBoost.")
    image = Image.open('feature_importance_plot.png')
    st.image(image, caption='Feature Importance Plot', use_column_width=True)
    st.write("-")

    st.write("### 4. Проверка точности и загрузка результатов на [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)")
    st.write("471 место в Kaggle")
    st.write("Результат на Kaggle: 0.12355")

   

    st.write("## Команда")
    st.write("- [Igor Svilanovich](https://github.com/himimori)")
    st.write("- [Andrei Miroshnichenko](https://github.com/Andriano2323)")
    st.write("- [Saiyyna Antonova](https://github.com/SainaAntonova)")
    


    

# Главная часть приложения
st.sidebar.title('Выберите страницу')
page = st.sidebar.radio("Навигация", ["Главная", "Прогноз", "Анализ данных"])


if page == "Главная":
    show_main_info()


elif page == "Прогноз":
    st.title('Прогноз цен на жилье')
    st.write("Добро пожаловать в приложение для прогнозирования цен на жилье!")
    st.write("Это приложение позволяет вам прогнозировать цены на жилье с использованием заранее обученной модели машинного обучения.")
    st.write("Вы также можете исследовать данные и просматривать метрики прогнозирования.")

    uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Данные файла:")
        st.write(data.head())
        
        prediction = predict(data)
        st.write("Прогноз:")
        st.write(prediction)


        # Возможность отображения метрик
        show_metrics = st.checkbox("Отображать метрики")
        if show_metrics:
            y_true = st.text_input("Введите y_true", value="")
            y_true = np.array([float(x.strip()) for x in y_true.split(",") if x.strip()])
            rmse, mae, r2 = compute_metrics(y_true, prediction)
            st.write("RMSE:", rmse)
            st.write("MAE:", mae)
            st.write("R2 Score:", r2)

elif page == "Анализ данных":
    st.title('Анализ данных')
    st.write("Добро пожаловать в приложение для прогнозирования цен на жилье!")
    st.write("Это приложение позволяет вам прогнозировать цены на жилье с использованием заранее обученной модели машинного обучения.")
    st.write("Вы также можете исследовать данные и просматривать метрики прогнозирования.")
    uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Данные файла:")
        st.write(data.head())
        st.write(data.describe().T)

        # Возможность отображения графиков
        plot_options = ["Bar plot", "Scatter plot", "Histogram", "Box plot", 'Heatmap']
        selected_plot = st.sidebar.selectbox("Выберите график", plot_options)
        
        if selected_plot == "Bar plot":
            x_axis = st.sidebar.selectbox("Выберите x-axis", data.columns)
            y_axis = st.sidebar.selectbox("Выберите y-axis", data.columns)
            st.write("Bar plot:")
            fig, ax = plt.subplots()
            sns.barplot(x=data[x_axis], y=data[y_axis], ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")  
            st.pyplot(fig)
            
        elif selected_plot == "Scatter plot":
            x_axis = st.sidebar.selectbox("Выберите x-axis", data.columns)
            y_axis = st.sidebar.selectbox("Выберите y-axis", data.columns)
            st.write("Scatter plot:")
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
            st.pyplot(fig)
            
        elif selected_plot == "Histogram":
            column = st.sidebar.selectbox("Выберите колонку", data.columns)
        
            st.write("Histogram:")
            fig, ax = plt.subplots()
            sns.histplot(data[column], ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Box plot":
            x_axis = st.sidebar.selectbox("Выберите x-axis", data.columns)
            y_axis = st.sidebar.selectbox("Выберите y-axis", data.columns)
            st.write("Box plot:")
            fig, ax = plt.subplots()
            sns.boxplot(x=data[x_axis], y=data[y_axis], ax=ax)
            st.pyplot(fig)
        elif selected_plot == 'Heatmap':
            numeric_data = data.select_dtypes(include=[np.number])
            fig = px.imshow(numeric_data.corr())
            fig.update_layout(title="Тепловая карта", xaxis_title="Признаки", yaxis_title="Признаки")
            st.plotly_chart(fig)
