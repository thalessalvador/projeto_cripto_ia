# -*- coding: utf-8 -*-
"""
Framework para Treinamento e Avaliação de Modelos de Regressão.

Este módulo fornece um conjunto de ferramentas para treinar, avaliar, comparar
e salvar múltiplos modelos de regressão utilizando a biblioteca scikit-learn.
Ele foi projetado para automatizar o pipeline de modelagem para problemas de
previsão de séries temporais financeiras.

Funcionalidades Principais:
-   **Suporte a Múltiplos Modelos:** Treina e avalia Regressão Linear,
    Regressão Polinomial, Random Forest e Redes Neurais (MLP Regressor).
-   **Validação Robusta:** Utiliza validação cruzada K-fold (no caso alterado pra TimeSeriesSplit) para obter
    métricas de desempenho mais estáveis e um conjunto de hold-out para
    uma avaliação final imparcial.
-   **Avaliação Abrangente:** Calcula e reporta múltiplas métricas, incluindo
    Mean Squared Error (MSE), Mean Absolute Error (MAE) e R-squared (R2).
-   **Comparação e Análise:** Gera tabelas comparativas de desempenho,
    identifica o melhor modelo com base no MSE e fornece visualizações
    (diagramas de dispersão) e análise de coeficientes para modelos lineares.
-   **Persistência de Modelos:** Salva os modelos treinados em disco usando
    joblib para uso futuro.
"""
import pandas as pd
import numpy as np
import logging
import os
import joblib  # type: ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit  # type: ignore
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline, Pipeline  # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # type: ignore


def train_and_evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,  # type: ignore
    model_type: str,
    kfolds: int,
    pair_name: str,
    models_folder: str,
    poly_degree: int = 2,
    n_estimators: int = 150,
    test_size: float = 0.3,
):
    """
    Treina, avalia e salva um único modelo de regressão usando K-fold (TimeSeriesSplit) e hold-out.

    Esta função executa um pipeline completo para um tipo de modelo especificado.
    Ela realiza a validação cruzada K-fold (TimeSeriesSplit) no conjunto de treino e, em seguida,
    avalia o desempenho em um conjunto de validação final (hold-out). Por fim,
    retreina o modelo com todos os dados e o salva em disco.

    Args:
        X (pd.DataFrame): DataFrame com as features (variáveis independentes).
        y (pd.Series): Series com a variável alvo (variável dependente).
        model_type (str): O tipo de modelo a ser treinado. Opções: 'MLP',
                          'Linear', 'Polynomial', 'RandomForest'.
        kfolds (int): O número de folds para a validação cruzada.
        pair_name (str): O nome do par de moedas, usado para nomear os arquivos.
        models_folder (str): O diretório para salvar o modelo treinado.
        poly_degree (int, optional): Grau a ser usado na Regressão Polinomial. Padrão é 2.
        n_estimators (int, optional): Número de árvores no RandomForest. Padrão é 150.
        test_size (float, optional): Proporção do dataset a ser usada como
                                     conjunto de validação (hold-out). Padrão é 0.3.

    Side Effects:
        - Salva o modelo treinado como um arquivo .pkl no `models_folder`.
        - Registra métricas detalhadas de desempenho no log.
    """
    logging.info(
        f"Iniciando treino e avaliação do modelo {model_type} para {pair_name}..."
    )

    # MLP com regularização alpha, early_stopoing evita sobreajuste no fim do treino, validation_fraction=0.2 ( quanto do conjunto de treino será separado internamente como validação durante o early_stopping)
    # Polynomial interaction_only reduz número de termos combinatórios (sempre usar StandadrdScaler antes)
    # RandomForest max_depht=5, min_samples_leaf=10 Limita a profundidade e exige folhas maiores, max_features="sqrt" evita sobreajuste e reduz correlação entre árvores
    model_mapping = {  # type: ignore
        "MLP": MLPRegressor(
            hidden_layer_sizes=(100, 50),
            alpha=0.001,  # Regularização L2
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50,
        ),
        "Linear": LinearRegression(),
        "Polynomial": make_pipeline(
            PolynomialFeatures(degree=poly_degree, interaction_only=True),
            LinearRegression(),
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
        ),
    }

    if model_type not in model_mapping:
        logging.error(f"Tipo de modelo '{model_type}' não suportado.")
        return

    if model_type == "Polynomial" and not (2 <= poly_degree <= 10):
        logging.error(
            f"Grau polinomial inválido: {poly_degree}. Deve estar entre 2 e 10."
        )
        return

    model = model_mapping[model_type]  # type: ignore
    # alterada a função Kfold  para TimeSeriesSplit pois para séries temporais elar respeita a ordem dos dados
    kf = TimeSeriesSplit(n_splits=kfolds)

    # aplicar_lag é uma função que aplica defasagem (lag) de 1 dia em todas as features para evitar vazamento de dados
    # e alinha a variável alvo (target) para o dia T, exceto a primeira linha
    # Isso é necessário para evitar que o modelo use informações futuras ao prever o preço de fechamento
    X_reset, y_reset = X.reset_index(drop=True), y.reset_index(drop=True)  # type: ignore
    X_reset, y_reset = aplicar_lag(X_reset, y_reset)  # type: ignore

    # Separa dados para validação final (hold-out) apenas se test_size > 0. A separação é feita levando em conta separação temporal
    if test_size > 0:
        split_index = int(len(X_reset) * (1 - test_size))
        X_train_full, X_val = X_reset.iloc[:split_index], X_reset.iloc[split_index:]
        y_train_full, y_val = y_reset.iloc[:split_index], y_reset.iloc[split_index:]  # type: ignore

    else:
        X_train_full, y_train_full = X_reset, y_reset  # type: ignore
        X_val, y_val = None, None

    if len(X_reset) < kfolds:
        logging.warning(
            f"Dados insuficientes para {kfolds}-Fold CV em {pair_name}. Treinando no conjunto completo."
        )
        if len(X_reset) > 0:
            try:
                model.fit(X_reset, y_reset)  # type: ignore
                model_filename = os.path.join(
                    models_folder,
                    f"{model_type.lower()}_{pair_name.replace(' ', '_')}.pkl",
                )
                joblib.dump({"model": model, "features": X_reset.columns.tolist()}, model_filename)  # type: ignore
                logging.info(
                    f"Modelo {model_type} para {pair_name} salvo em: {model_filename}"
                )
            except Exception as e:
                logging.error(
                    f"Erro ao salvar modelo {model_type} para {pair_name}: {e}"
                )
        return

    mse_scores = np.zeros(kfolds)
    mae_scores = np.zeros(kfolds)
    r2_scores = np.zeros(kfolds)

    for i, (train_index, test_index) in enumerate(kf.split(X_train_full)):  # type: ignore
        X_train, X_test = X_train_full.iloc[train_index], X_train_full.iloc[test_index]  # type: ignore
        y_train, y_test = y_train_full.iloc[train_index], y_train_full.iloc[test_index]  # type: ignore

        try:
            model.fit(X_train, y_train)  # type: ignore
            y_pred = model.predict(X_test)  # type: ignore
            mse_scores[i] = mean_squared_error(y_test, y_pred)  # type: ignore
            mae_scores[i] = mean_absolute_error(y_test, y_pred)  # type: ignore
            r2_scores[i] = r2_score(y_test, y_pred)  # type: ignore
            logging.info(
                f"  Fold {i+1}: MSE={mse_scores[i]:.4f}, MAE={mae_scores[i]:.4f}, R2={r2_scores[i]:.4f}"
            )
        except Exception as e:
            logging.error(f"Erro no Fold {i+1} para {model_type}: {e}")
            mse_scores[i], mae_scores[i], r2_scores[i] = np.nan, np.nan, np.nan

    if not np.isnan(mse_scores).all():
        logging.info(f"Resultados Médios para {model_type} ({kfolds}-Fold CV):")
        logging.info(f"  MSE Médio: {np.nanmean(mse_scores):.4f}")
        logging.info(f"  MAE Médio: {np.nanmean(mae_scores):.4f}")
        logging.info(f"  R2 Médio: {np.nanmean(r2_scores):.4f}")

        if test_size > 0 and X_val is not None:
            try:
                model.fit(X_train_full, y_train_full)  # type: ignore
                y_pred_val = model.predict(X_val)  # type: ignore
                final_r2 = r2_score(y_val, y_pred_val)  # type: ignore
                final_mae = mean_absolute_error(y_val, y_pred_val)  # type: ignore
                final_mse = mean_squared_error(y_val, y_pred_val)  # type: ignore
                logging.info(
                    f"[{pair_name}] Hold-Out - R2: {final_r2:.4f}, MAE: {final_mae:.2f}, MSE: {final_mse:.2f}"
                )
            except Exception as e:
                logging.error(
                    f"Erro ao avaliar no conjunto hold-out para {model_type} em {pair_name}: {e}"
                )

        try:
            model.fit(X_reset, y_reset)  # type: ignore
            model_filename = os.path.join(
                models_folder, f"{model_type.lower()}_{pair_name.replace(' ', '_')}.pkl"
            )
            joblib.dump({"model": model, "features": X_reset.columns.tolist()}, model_filename)  # type: ignore
            logging.info(f"Modelo final {model_type} salvo em: {model_filename}")
        except Exception as e:
            logging.error(f"Erro ao salvar modelo final {model_type}: {e}")
    else:
        logging.warning(f"Nenhum score foi calculado para {model_type} em {pair_name}.")


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,  # type: ignore
    kfolds: int,
    pair_name: str,
    plots_folder: str,
    poly_degree: int = 2,
    n_estimators: int = 150,
    test_size: float = 0.3,
):
    """
    Compara múltiplos modelos de regressão, exibe métricas e gera gráficos.

    Executa a validação cruzada K-fold (TimeSeriesSplit) para vários modelos, registra uma tabela
    comparativa de desempenho, identifica o melhor regressor com base no MSE
    médio e gera visualizações para análise.

    Args:
        X (pd.DataFrame): DataFrame com as features de entrada.
        y (pd.Series): Series com a variável alvo.
        kfolds (int): O número de folds para a validação cruzada.
        pair_name (str): O nome do par de moedas para nomear os arquivos.
        plots_folder (str): O diretório para salvar os gráficos de comparação.
        poly_degree (int, optional): Grau para a Regressão Polinomial. Padrão é 2.
        n_estimators (int, optional): Número de árvores no RandomForest. Padrão é 150.
        test_size (float, optional): Proporção para o conjunto de hold-out. Padrão é 0.3.

    Side Effects:
        - Registra uma tabela de comparação de modelos no log.
        - Salva um diagrama de dispersão comparativo em `plots_folder`.
        - Registra a análise de coeficientes para modelos lineares no log.
    """
    logging.info(f"Comparando modelos para {pair_name}...")

    # MLP com regularização alpha, early_stopoing evita sobreajuste no fim do treino, validation_fraction=0.2 ( quanto do conjunto de treino será separado internamente como validação durante o early_stopping)
    # Polynomial interaction_only reduz número de termos combinatórios (sempre usar StandadrdScaler antes)
    # RandomForest max_depht=5, min_samples_leaf=10 Limita a profundidade e exige folhas maiores, max_features="sqrt" evita sobreajuste e reduz correlação entre árvores
    models = {  # type: ignore
        "MLP": MLPRegressor(
            hidden_layer_sizes=(100, 50),
            alpha=0.001,  # Regularização L2
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50,
        ),
        "Linear": LinearRegression(),
        "Polynomial": make_pipeline(
            PolynomialFeatures(degree=poly_degree, interaction_only=True),
            LinearRegression(),
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
        ),
    }

    results = []
    # alterada a função Kfold  para TimeSeriesSplit pois para séries temporais elar respeita a ordem dos dados
    kf = TimeSeriesSplit(n_splits=kfolds)

    # aplicar_lag é uma função que aplica defasagem (lag) de 1 dia em todas as features para evitar vazamento de dados
    # e alinha a variável alvo (target) para o dia T, exceto a primeira linha
    # Isso é necessário para evitar que o modelo use informações futuras ao prever o preço de fechamento
    X_reset, y_reset = X.reset_index(drop=True), y.reset_index(drop=True)  # type: ignore
    X_reset, y_reset = aplicar_lag(X_reset, y_reset)  # type: ignore

    # Separa os dados para validação real (hold-out), levando em conta a temporalidade e não aleatoriedade
    if test_size > 0:
        split_index = int(len(X_reset) * (1 - test_size))
        X_train_full, X_val = X_reset.iloc[:split_index], X_reset.iloc[split_index:]
        y_train_full, y_val = y_reset.iloc[:split_index], y_reset.iloc[split_index:]  # type: ignore

    else:
        X_train_full, y_train_full = X_reset, y_reset  # type: ignore
        X_val, y_val = None, None

    if len(X_reset) < kfolds:
        logging.warning(
            f"Dados insuficientes para {kfolds}-Fold CV para comparação em {pair_name}."
        )
        return

    resultadosHoldOut = "\n"
    for model_name, model in models.items():  # type: ignore
        mse_scores = np.zeros(kfolds)
        mae_scores = np.zeros(kfolds)
        r2_scores = np.zeros(kfolds)
        std_error_scores = np.zeros(kfolds)

        for i, (train_index, test_index) in enumerate(kf.split(X_train_full)):  # type: ignore
            X_train, X_test = X_train_full.iloc[train_index], X_train_full.iloc[test_index]  # type: ignore
            y_train, y_test = y_train_full.iloc[train_index], y_train_full.iloc[test_index]  # type: ignore

            try:
                model.fit(X_train, y_train)  # type: ignore
                y_pred = model.predict(X_test)  # type: ignore
                mse_scores[i] = mean_squared_error(y_test, y_pred)  # type: ignore
                mae_scores[i] = mean_absolute_error(y_test, y_pred)  # type: ignore
                r2_scores[i] = r2_score(y_test, y_pred)  # type: ignore
                std_error_scores[i] = np.std(y_test - y_pred)  # type: ignore
            except Exception as e:
                logging.error(
                    f"Erro na comparação do modelo {model_name} no Fold {i+1}: {e}"
                )
                mse_scores[i], mae_scores[i], r2_scores[i], std_error_scores[i] = (
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )

        # Avaliação no conjunto de validação final (hold-out)
        if test_size > 0 and X_val is not None:
            try:
                model.fit(X_train_full, y_train_full)  # type: ignore
                y_val_pred = model.predict(X_val)  # type: ignore
                holdout_r2 = r2_score(y_val, y_val_pred)  # type: ignore
                holdout_mae = mean_absolute_error(y_val, y_val_pred)  # type: ignore
                holdout_mse = mean_squared_error(y_val, y_val_pred)  # type: ignore
                resultadosHoldOut += f"[{pair_name}] {model_name} - Hold-Out -> R2: {holdout_r2:.4f}, MAE: {holdout_mae:.2f}, MSE: {holdout_mse:.2f}\n"
            except Exception as e:
                logging.error(
                    f"Erro na avaliação final (hold-out) do modelo {model_name}: {e}"
                )

        if not np.isnan(mse_scores).all():
            results.append(  # type: ignore
                {
                    "Model": model_name,
                    "Avg MSE": np.nanmean(mse_scores),
                    "Avg MAE": np.nanmean(mae_scores),
                    "Avg R2": np.nanmean(r2_scores),
                    "Avg Std Error": np.nanmean(std_error_scores),
                }
            )

    if not results:
        logging.warning(f"Nenhum resultado de modelo foi gerado para {pair_name}.")
        return

    df_results = pd.DataFrame(results)
    logging.info(
        f"\n*** Comparação de Modelos para {pair_name} ***\n{df_results.to_string()}"  # type: ignore
    )
    logging.info(resultadosHoldOut)

    if test_size > 0 and X_val is not None:
        # Plota o gráfico de dispersão para o conjunto de validação final, Ou seja, como o modelo se comparta com os dados de hold-out
        plot_scatter_holdout(models, X_val, y_val, pair_name, plots_folder)  # type: ignore

    best_regressor = df_results.loc[df_results["Avg MSE"].idxmin()]  # type: ignore
    logging.info(
        f"Melhor Regressor téorico (Hold-out) para {pair_name} (baseado em MSE): {best_regressor['Model']}."
    )
    logging.info(
        f"Essa informação é a título de curiosidade, ou seja, nesses dados específicos qual seria o melhor regressor. Entretanto, o modelo mais generalista não é esse, mas o já apontado e gravado."
    )

    # Cálculo do erro padrão entre o MLP e o melhor regressor
    mlp_row = df_results[df_results["Model"] == "MLP"]
    if not mlp_row.empty and best_regressor["Model"] != "MLP":  # type: ignore
        std_err_mlp = mlp_row["Avg Std Error"].iloc[0]  # type: ignore # type: ignore
        std_err_best = best_regressor["Avg Std Error"]  # type: ignore
        logging.info(f"Erro Padrão do MLP: {std_err_mlp:.4f}")
        logging.info(
            f"Erro Padrão do Melhor Regressor ({best_regressor['Model']}): {std_err_best:.4f}"
        )
        logging.info(
            f"Diferença no Erro Padrão (MLP vs Melhor): {abs(std_err_mlp - std_err_best):.4f}"  # type: ignore
        )

    _plot_scatter_comparison(X_reset, y_reset, models, pair_name, plots_folder)
    _log_coefficients(X_reset, y_reset, models, pair_name)


def _plot_scatter_comparison(X, y, models, pair_name, plots_folder):  # type: ignore
    """
    Plota um diagrama de dispersão comparando valores reais vs. previstos.

    Para cada modelo fornecido, a função treina o modelo no conjunto de dados
    completo, faz previsões e plota os resultados em um único gráfico de dispersão.
    Inclui uma linha de referência ideal (y=x).

    Args:
        X (pd.DataFrame): DataFrame de features.
        y (pd.Series): Series da variável alvo.
        models (dict): Dicionário de modelos a serem plotados.
        pair_name (str): Nome do par de moedas para o título do gráfico.
        plots_folder (str): Pasta para salvar a imagem do gráfico.

    Side Effects:
        - Salva um arquivo de imagem .png no `plots_folder`.
    """
    plt.figure(figsize=(12, 8))  # type: ignore
    sns.set_palette("viridis")
    for model_name, model in models.items():  # type: ignore
        try:
            model.fit(X, y)  # type: ignore
            y_pred = model.predict(X)  # type: ignore # type: ignore
            plt.scatter(  # type: ignore
                y,  # type: ignore
                y_pred,  # type: ignore
                alpha=0.6,
                label=f"{model_name} (R2: {r2_score(y, y_pred):.4f})",  # type: ignore
            )
        except Exception as e:
            logging.error(
                f"Erro ao gerar previsões para dispersão do modelo {model_name}: {e}"
            )

    plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2, label="Linha Ideal")  # type: ignore
    plt.xlabel("Preço Real (USDT)")  # type: ignore
    plt.ylabel("Preço Previsto (USDT)")  # type: ignore
    plt.title(f"Diagrama de Dispersão: Real vs. Previsto para {pair_name}")  # type: ignore
    plt.legend()  # type: ignore
    plt.grid(True)  # type: ignore
    plot_path = os.path.join(
        plots_folder, f"scatter_plot_models_{pair_name.replace(' ', '_')}.png"  # type: ignore
    )
    plt.savefig(plot_path, dpi=150)  # type: ignore
    plt.close()
    logging.info(f"Diagrama de dispersão salvo em: {plot_path}")


def plot_scatter_holdout(models, X_val, y_val, pair_name, plots_folder):  # type: ignore
    """
    Gera um gráfico de dispersão usando o modelo escolhido no treino em k-fold (TimeSeriesSplit) para os dados de hold-out (validação final).
    Avalia o desempenho do modelo em relação aos valores reais do conjunto de validação.

    Args:
        models (dict): Dicionário com nome e instância de modelos treinados.
        X_val (pd.DataFrame): Conjunto de validação (features).
        y_val (pd.Series): Conjunto de validação (valores reais).
        pair_name (str): Nome do par de moedas (ex: "BTC_USDT").
        plots_folder (str): Caminho onde salvar o gráfico gerado.
    """
    try:
        plt.figure(figsize=(12, 8))  # type: ignore
        sns.set_palette("Set2")

        for model_name, model in models.items():  # type: ignore
            try:
                y_pred = model.predict(X_val)  # type: ignore
                plt.scatter(  # type: ignore
                    y_val,  # type: ignore
                    y_pred,  # type: ignore
                    alpha=0.6,
                    label=f"{model_name} (R2: {r2_score(y_val, y_pred):.4f})",  # type: ignore
                )
            except Exception as e:
                logging.warning(f"[Hold-Out Scatter] Falha para {model_name}: {e}")

        plt.plot(  # type: ignore
            [y_val.min(), y_val.max()],  # type: ignore
            [y_val.min(), y_val.max()],  # type: ignore
            "k--",
            lw=2,
            label="Linha Ideal",
        )
        plt.xlabel("Preço Real (USDT)")  # type: ignore
        plt.ylabel("Preço Previsto (USDT)")  # type: ignore
        plt.title(  # type: ignore
            f"Dispersão Hold-Out (Dados que o modelo não conhecia): Real vs. Previsto ({pair_name})"
        )
        plt.legend()  # type: ignore
        plt.grid(True)  # type: ignore

        if not os.path.exists(plots_folder):  # type: ignore # type: ignore
            os.makedirs(plots_folder)  # type: ignore

        plot_path = os.path.join(
            plots_folder, f"scatter_holdout_{pair_name.replace(' ', '_')}.png"  # type: ignore
        )
        plt.savefig(plot_path, dpi=150)  # type: ignore
        plt.close()
        logging.info(f"Gráfico de dispersão (hold-out) salvo em: {plot_path}")
    except Exception as e:
        logging.error(
            f"Erro ao gerar gráfico de dispersão hold-out para {pair_name}: {e}"
        )


def _log_coefficients(X, y, models, pair_name):  # type: ignore
    """
    Registra os coeficientes e a equação para modelos lineares e polinomiais.

    Esta função itera sobre os modelos e, para aqueles que são lineares ou
    polinomiais, extrai seus coeficientes e intercepto para formar e registrar
    a equação matemática do modelo.

    Args:
        X (pd.DataFrame): DataFrame de features.
        y (pd.Series): Series da variável alvo.
        models (dict): Dicionário de modelos para análise.
        pair_name (str): Nome do par de moedas para o log.

    Side Effects:
        - Imprime a equação do modelo no log de informações.
    """
    logging.info(f"Análise de Coeficientes e Equações para {pair_name}:")
    for model_name, model in models.items():  # type: ignore
        if "Linear" not in model_name and "Polynomial" not in model_name:
            logging.info(f"  Modelo: {model_name} -> Não é uma equação linear simples.")
            continue
        try:
            model.fit(X, y)  # type: ignore
            if isinstance(model, Pipeline):
                linear_model = model.named_steps["linearregression"]  # type: ignore
                poly_features = model.named_steps["polynomialfeatures"]  # type: ignore
                feature_names = poly_features.get_feature_names_out(X.columns)  # type: ignore
                intercept = linear_model.intercept_  # type: ignore
                coefs = linear_model.coef_  # type: ignore
            else:
                intercept = model.intercept_  # type: ignore
                coefs = model.coef_  # type: ignore
                feature_names = X.columns  # type: ignore # type: ignore

            equation = f"y = {intercept:.4f}"
            for feature, coef in zip(feature_names, coefs):  # type: ignore
                equation += f" + ({coef:.4f} * {feature})"
            logging.info(f"  Modelo: {model_name} -> {equation}\n")
        except Exception as e:
            logging.warning(
                f"Não foi possível determinar a equação para {model_name}: {e}"
            )


def get_best_model_by_mse(  # type: ignore
    X: pd.DataFrame,
    y: pd.Series,  # type: ignore
    kfolds: int,
    poly_degree: int = 2,
    n_estimators: int = 150,
    test_size: float = 0.3,
):
    """
    Seleciona o melhor modelo de regressão com base em validação cruzada (K-Fold (TimeSeriesSplit)) e avalia seu desempenho em um conjunto de validação (hold-out).

    Esta função realiza os seguintes passos:
    1. Separa os dados em treino e validação final (hold-out), usando `test_size`.
    2. Executa validação cruzada K-Fold (TimeSeriesSplit) apenas no conjunto de treino para estimar o desempenho médio (MSE) de cada modelo.
    3. Identifica o modelo com menor MSE médio.
    4. Reajusta o modelo vencedor em todo o conjunto de treino e avalia seu desempenho final nos dados de validação (hold-out), apenas para fins de comparação.
    5. Retorna o modelo final treinado com os dados de treino e seu nome.

    Modelos considerados:
    - Regressão Linear
    - Regressão Polinomial (grau configurável)
    - Random Forest
    - MLP Regressor (rede neural multicamadas)

    Parâmetros:
        X (pd.DataFrame): Conjunto de variáveis independentes (features).
        y (pd.Series): Variável alvo (preço de fechamento).
        kfolds (int): Número de divisões para validação cruzada.
        poly_degree (int, opcional): Grau máximo da Regressão Polinomial. Padrão: 2.
        n_estimators (int, opcional): Número de árvores no Random Forest. Padrão: 150.
        test_size (float, opcional): Proporção dos dados a serem reservados para o hold-out. Padrão: 0.3.

    Retorna:
        Tuple[RegressorMixin, str]: O modelo treinado (ajustado com os dados de treino) e o nome do modelo.

    Efeitos colaterais:
        - Registra no log os resultados de desempenho médio (K-Fold (TimeSeriesSplit)) e final (hold-out) para cada modelo.
    """

    logging.info(
        "Executando seleção automática do melhor modelo com base em MSE (com hold-out)..."
    )
    # MLP com regularização alpha, early_stopoing evita sobreajuste no fim do treino, validation_fraction=0.2 ( quanto do conjunto de treino será separado internamente como validação durante o early_stopping)
    # Polynomial interaction_only reduz número de termos combinatórios (sempre usar StandadrdScaler antes)
    # RandomForest max_depht=5, min_samples_leaf=10 Limita a profundidade e exige folhas maiores, max_features="sqrt" evita sobreajuste e reduz correlação entre árvores
    model_defs = {  # type: ignore
        "MLP": MLPRegressor(
            hidden_layer_sizes=(100, 50),
            alpha=0.001,  # Regularização L2
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50,
        ),
        "Linear": LinearRegression(),
        "Polynomial": make_pipeline(
            PolynomialFeatures(degree=poly_degree, interaction_only=True),
            LinearRegression(),
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
        ),
    }
    # aplicar_lag é uma função que aplica defasagem (lag) de 1 dia em todas as features para evitar vazamento de dados
    # e alinha a variável alvo (target) para o dia T, exceto a primeira linha
    # Isso é necessário para evitar que o modelo use informações futuras ao prever o preço de fechamento
    X_reset, y_reset = X.reset_index(drop=True), y.reset_index(drop=True)  # type: ignore
    X_reset, y_reset = aplicar_lag(X_reset, y_reset)  # type: ignore

    # Divide os dados entre treino e validação (hold-out) levando em conta a separação temporal
    if test_size > 0:
        split_index = int(len(X_reset) * (1 - test_size))
        X_train_full, X_val = X_reset.iloc[:split_index], X_reset.iloc[split_index:]
        y_train_full, y_val = y_reset.iloc[:split_index], y_reset.iloc[split_index:]  # type: ignore

    else:
        X_train_full, y_train_full = X_reset, y_reset  # type: ignore
        X_val, y_val = None, None

    best_model = None
    best_name = None
    best_mse = float("inf")

    # alterada a função Kfold  para TimeSeriesSplit pois para séries temporais elar respeita a ordem dos dados
    kf = TimeSeriesSplit(n_splits=kfolds)

    for name, model in model_defs.items():  # type: ignore
        try:
            mse_scores = []
            for train_idx, test_idx in kf.split(X_train_full):  # type: ignore
                X_train, X_test = X_train_full.iloc[train_idx], X_train_full.iloc[test_idx]  # type: ignore
                y_train, y_test = y_train_full.iloc[train_idx], y_train_full.iloc[test_idx]  # type: ignore
                model.fit(X_train, y_train)  # type: ignore
                y_pred = model.predict(X_test)  # type: ignore
                mse_scores.append(mean_squared_error(y_test, y_pred))  # type: ignore

            avg_mse = np.mean(mse_scores)  # type: ignore
            logging.info(
                f"[{name}] MSE Médio (K-Fold (TimeSeriesSplit)): {avg_mse:.4f}"
            )

            if avg_mse < best_mse:
                best_mse = avg_mse
                best_model = model  # type: ignore
                best_name = name
        except Exception as e:
            logging.error(f"Erro ao avaliar modelo {name}: {e}")

    # Reavalia o melhor modelo no conjunto de hold-out (apenas para log)
    if best_model is not None and X_val is not None:
        try:
            best_model.fit(X_train_full, y_train_full)  # type: ignore
            y_pred_val = best_model.predict(X_val)  # type: ignore
            val_mse = mean_squared_error(y_val, y_pred_val)  # type: ignore
            val_mae = mean_absolute_error(y_val, y_pred_val)  # type: ignore
            val_r2 = r2_score(y_val, y_pred_val)  # type: ignore
            logging.info(
                f"[{best_name}] Avaliação no Hold-Out → MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}"
            )
        except Exception as e:
            logging.warning(
                f"Erro ao avaliar o melhor modelo ({best_name}) no hold-out: {e}"
            )

    return best_model, best_name  # type: ignore


def limpar_modelos_antigos(pair_name: str, models_folder: str):
    """
    Remove modelos antigos (.pkl) do par especificado antes de salvar um novo.

    Esta função busca por arquivos de modelos existentes associados ao par de
    moedas (`pair_name`) dentro da pasta de modelos (`models_folder`) e remove
    todos, exceto o que será salvo em seguida. Isso evita múltiplos modelos
    conflitantes no diretório e garante que apenas o modelo atual esteja disponível
    para etapas posteriores como simulação de lucro ou predição.

    Args:
        pair_name (str): Nome do par de moedas (ex: 'BTC_USDT').
        models_folder (str): Caminho para a pasta onde os modelos são armazenados.

    Side Effects:
        - Remove arquivos `.pkl` do disco para os modelos: MLP, Linear, Polynomial e Random Forest.
        - Registra os arquivos removidos no log.
    """
    formatos = ["mlp", "linear", "polynomial", "randomforest"]
    for prefix in formatos:
        caminho = os.path.join(models_folder, f"{prefix}_{pair_name}.pkl")
        if os.path.exists(caminho):
            os.remove(caminho)
            logging.info(f"Modelo antigo removido: {caminho}")


def aplicar_lag(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:  # type: ignore
    """
    Aplica defasagem (lag) de 1 dia em todas as features para evitar vazamento de dados.

    Args:
        X (pd.DataFrame): Conjunto de features (com dados do dia T).
        y (pd.Series): Variável alvo (preço de fechamento do dia T).

    Returns:
        Tuple com:
            - X_lagged (features do dia T-1)
            - y_alinhado (target do dia T, exceto a primeira linha)
    """
    X_lagged = X.shift(1)
    X_lagged = X_lagged.iloc[1:]
    y_alinhado = y.iloc[1:]  # type: ignore
    return X_lagged, y_alinhado  # type: ignore
