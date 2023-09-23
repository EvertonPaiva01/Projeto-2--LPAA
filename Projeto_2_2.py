# Importações
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


def load_and_preprocess_data():
    # Carregando o dataset de diabetes
    diabetes = datasets.load_diabetes()

    # Convertendo o dataset para um DataFrame do Pandas
    df = pd.DataFrame(data=np.c_[diabetes['data'], diabetes['target']],
                      columns=diabetes['feature_names'] + ['target'])
    return df


def display_basic_info(df):
    # Exibindo as primeiras linhas do DataFrame
    print(df.head())
    # Exibindo as estatísticas descritivas do DataFrame
    print(df.describe())


def plot_correlation_matrix(df):
    # Calculando a matriz de correlação
    correlation_matrix = df.corr()

    # Configurando o tamanho da figura
    plt.figure(figsize=(12, 8))

    # Criando um mapa de calor para visualizar a correlação entre as variáveis
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação')
    plt.show()


def train_test_splitting(df):
    X = df.drop(columns=['target'])
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def define_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regression": DecisionTreeRegressor(random_state=42),
        "Ridge Regression": Ridge(random_state=42)
    }


def train_and_evaluate(models, X_train, y_train, folds=10):
    model_results = {}
    for model_name, model in models.items():
        rmse = sqrt(abs(cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=folds).mean()))
        mae = abs(cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=folds).mean())
        r2 = cross_val_score(model, X_train, y_train, scoring='r2', cv=folds).mean()
        model_results[model_name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    return model_results


def plot_scatter(models, X_train, y_train, X_test, y_test):
    plt.figure(figsize=(15, 5))
    for i, (model_name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        plt.subplot(1, 3, i + 1)
        plt.scatter(y_test, predictions)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.title(f'{model_name} - Scatter Plot')
        plt.xlabel('Valores Reais')
        plt.ylabel('Previsões')
    plt.tight_layout()
    plt.show()


def plot_residuals(models, X_train, y_train, X_test, y_test):
    plt.figure(figsize=(15, 5))
    for i, (model_name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        residuals = y_test - predictions
        plt.subplot(1, 3, i + 1)
        plt.scatter(y_test, residuals)
        plt.axhline(0, color='k', linestyle='--', lw=2)
        plt.title(f'{model_name} - Residual Plot')
        plt.xlabel('Valores Reais')
        plt.ylabel('Resíduos')
    plt.tight_layout()
    plt.show()


def plot_learning_curves(models, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10)):
    plt.figure(figsize=(15, 5))
    for i, (model_name, model) in enumerate(models.items()):
        train_sizes, train_scores, validation_scores = learning_curve(model, X_train, y_train, train_sizes=train_sizes, cv=10, scoring='neg_mean_squared_error')
        train_scores_mean = -train_scores.mean(axis=1)
        train_scores_std = train_scores.std(axis=1)
        validation_scores_mean = -validation_scores.mean(axis=1)
        validation_scores_std = validation_scores.std(axis=1)
        plt.subplot(1, 3, i + 1)
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Score')
        plt.plot(train_sizes, validation_scores_mean, 'o-', color='g', label='Cross-Validation Score')
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
        plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha=0.1, color='g')
        plt.title(f'{model_name} - Learning Curve')
        plt.xlabel('Training Size')
        plt.ylabel('Mean Squared Error')
        plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_and_preprocess_data()
    display_basic_info(df)
    plot_correlation_matrix(df)

    X_train, X_test, y_train, y_test = train_test_splitting(df)
    models = define_models()

    results = train_and_evaluate(models, X_train, y_train)
    results_df = pd.DataFrame(results).T
    print(results_df)

    plot_scatter(models, X_train, y_train, X_test, y_test)
    plot_residuals(models, X_train, y_train, X_test, y_test)
    plot_learning_curves(models, X_train, y_train)
