"""
Mobile Price Classification Project
===================================
Este projeto utiliza dados de características de celulares para classificar a faixa de preço.
Dataset: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

class MobilePriceClassifier:
    """
    Classe principal para classificação de preços de celulares
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.results = {}
        
    def load_data(self):
        """Carrega os dados de treino e teste"""
        try:
            self.train_data = pd.read_csv('train.csv')
            self.test_data = pd.read_csv('test.csv')
            print(f"Dados carregados: {len(self.train_data)} treino, {len(self.test_data)} teste")
            return True
        except FileNotFoundError:
            print("Erro: Arquivos train.csv ou test.csv não encontrados")
            return False
    
    def explore_data(self):
        """Análise básica dos dados"""
        print("\nAnálise dos Dados")
        print(f"Shape: {self.train_data.shape}")
        print(f"Colunas: {list(self.train_data.columns)}")
        
        # Verificar se tem valores nulos OBS: é esperado que o dataset não possua valores nulos, esse código serve apenas para confirmar isso
        if self.train_data.isnull().sum().sum() > 0:
            print("Existem valores nulos nos dados")
        else:
            print("Nenhum valor nulo encontrado")
        
        # Distribuição do target
        print("\nDistribuição das classes:")
        print(self.train_data['price_range'].value_counts().sort_index())
    
    def prepare_data(self):
        """Prepara os dados para treinamento"""
        # Separar features e target
        X = self.train_data.drop(['price_range'], axis=1)
        y = self.train_data['price_range']
        
        # Dividir em treino e validação
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizaçao os dados
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        
        print(f"Dados preparados - Treino: {len(self.X_train)}, Validação: {len(self.X_val)}")
    
    def train_models(self):
        """Treina diferentes modelos"""
        
        # Definir os modelos
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        
        # Treinar cada modelo
        for name, model in models.items():
            print(f"Treinando {name}")
            
            if name in ['Logistic Regression', 'SVM']:
                model.fit(self.X_train_scaled, self.y_train)
                predictions = model.predict(self.X_val_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                predictions = model.predict(self.X_val)
            
            # Avaliar
            accuracy = accuracy_score(self.y_val, predictions)
            
            # Cross validation
            if name in ['Logistic Regression', 'SVM']:
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  Acurácia: {accuracy:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    def evaluate_models(self):
        """Avalia e compara os modelos"""
        print("\n=== Resultados dos Modelos ===")
        
        # Criar DataFrame com resultados
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        print(results_df.round(4))
        
        # Melhor modelo
        self.best_model_name = results_df.index[0]
        self.best_model = self.models[self.best_model_name]
        
        print(f"\nMelhor modelo: {self.best_model_name}")
        print(f"Acurácia: {results_df.loc[self.best_model_name, 'accuracy']:.4f}")
        
        if self.best_model_name in ['Logistic Regression', 'SVM']:
            best_predictions = self.best_model.predict(self.X_val_scaled)
        else:
            best_predictions = self.best_model.predict(self.X_val)
        
        print(f"\nRelatório do {self.best_model_name}:")
        print(classification_report(self.y_val, best_predictions))
    
    def make_predictions(self):
        """Faz predições no conjunto de teste"""
        print("\n=== Fazendo Predições ===")
        
        # Preparar dados de teste
        X_test = self.test_data.drop(['id'], axis=1)
        
        if self.best_model_name in ['Logistic Regression', 'SVM']:
            X_test_scaled = self.scaler.transform(X_test)
            predictions = self.best_model.predict(X_test_scaled)
        else:
            predictions = self.best_model.predict(X_test)
        
        # Criar arquivo de submissão
        submission = pd.DataFrame({
            'id': self.test_data['id'],
            'price_range': predictions
        })
        
        submission.to_csv('predictions.csv', index=False)
        print("Predições salvas em 'predictions.csv'")
        
        print("Distribuição das predições:")
        print(pd.Series(predictions).value_counts().sort_index())
    
    def create_simple_visualization(self):
        """Criar algumas visualizações básicas"""
        plt.figure(figsize=(15, 10))
        
        # 1. Distribuição do target
        plt.subplot(2, 3, 1)
        self.train_data['price_range'].value_counts().sort_index().plot(kind='bar')
        plt.title('Distribuição das Classes')
        plt.xlabel('Price Range')
        plt.ylabel('Quantidade')
        
        # 2. RAM vs Price Range
        plt.subplot(2, 3, 2)
        sns.boxplot(data=self.train_data, x='price_range', y='ram')
        plt.title('RAM por Faixa de Preço')
        
        # 3. Battery Power vs Price Range
        plt.subplot(2, 3, 3)
        sns.boxplot(data=self.train_data, x='price_range', y='battery_power')
        plt.title('Battery Power por Faixa de Preço')
        
        # 4. Correlação das principais features
        plt.subplot(2, 3, 4)
        important_features = ['ram', 'battery_power', 'px_width', 'px_height', 'price_range']
        correlation = self.train_data[important_features].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlação entre Features')
        
        # 5. Comparação dos modelos
        plt.subplot(2, 3, 5)
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        plt.bar(range(len(model_names)), accuracies)
        plt.xticks(range(len(model_names)), model_names, rotation=45)
        plt.title('Comparação dos Modelos')
        plt.ylabel('Acurácia')
        
        plt.tight_layout()
        plt.savefig('analysis_results.png', dpi=300, bbox_inches='tight')
        print("Gráficos salvos em 'analysis_results.png'")
        plt.show()

def main():
    classifier = MobilePriceClassifier()
    
    if not classifier.load_data():
        return
    
    classifier.explore_data()
    classifier.prepare_data()
    classifier.train_models()
    classifier.evaluate_models()
    classifier.make_predictions()
    classifier.create_simple_visualization()
    
    print("\nAnálise finalizada")

if __name__ == "__main__":
    main() 
