# Mobile Price Classification

Um projeto de classificação de preços de celulares usando Machine Learning.

## Sobre o Projeto

Este projeto classifica celulares em 4 faixas de preço baseado nas suas especificações técnicas:
- **Classe 0**: Baixo custo
- **Classe 1**: Custo médio  
- **Classe 2**: Alto custo
- **Classe 3**: Muito alto custo

**Observação** - é esperado que o dataset não possua valores nulos, o código que verifica serve apenas para confirmação

## Dataset

O dataset contém informações de 2000 celulares para treino e 1000 para teste, com as seguintes características:
- **battery_power**: Energia total da bateria (mAh)
- **blue**: Tem Bluetooth (0/1)
- **clock_speed**: Velocidade do processador
- **dual_sim**: Suporte a dual SIM (0/1)
- **fc**: Megapixels da câmera frontal
- **four_g**: Suporte a 4G (0/1)
- **int_memory**: Memória interna (GB)
- **m_dep**: Espessura do celular (cm)
- **mobile_wt**: Peso do celular (g)
- **n_cores**: Número de núcleos do processador
- **pc**: Megapixels da câmera principal
- **px_height**: Altura da resolução em pixels
- **px_width**: Largura da resolução em pixels
- **ram**: Memória RAM (MB)
- **sc_h**: Altura da tela (cm)
- **sc_w**: Largura da tela (cm)
- **talk_time**: Tempo de conversação (horas)
- **three_g**: Suporte a 3G (0/1)
- **touch_screen**: Tela touch (0/1)
- **wifi**: Suporte a WiFi (0/1)

## Como Usar

```bash
python mobile_price_classification.py
```

O script executa uma análise completa que inclui:
- Análise exploratória dos dados
- Treinamento de múltiplos algoritmos de ML
- Comparação de performance dos modelos
- Geração de visualizações
- Criação de predições finais

## Dependências

Instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

## Resultados

O projeto alcança até **96.5% de acurácia** usando Logistic Regression.

### Ranking dos Modelos:
1. **Logistic Regression**: 96.5%
2. **Gradient Boosting**: 92.0%
3. **Random Forest**: 88.0%
4. **Decision Tree**: 85.5%
5. **SVM**: 87.0%

### Features Mais Importantes:
1. **RAM**: Principal fator para classificação
2. **Battery Power**: Segundo mais importante
3. **Resolução da tela**: Também muito relevante

## Arquivos Gerados

- `predictions.csv`: Predições finais
- `analysis_results.png`: Gráficos de análise

## Estrutura do Projeto

```
├── train.csv              # Dados de treino
├── test.csv               # Dados de teste
├── mobile_price_classification.py  # Script principal
├── README.md              # Este arquivo
└── requirements.txt       # Dependências
```

## Próximos Passos

- Testar ensemble de modelos
- Adicionar mais features engineered
- Otimizar hiperparâmetros

## Autores

Projeto desenvolvido por Thiago Dias e Gabriel Lopes para resolver um problema de classificação.

