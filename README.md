# Sistemas Inteligentes para Bioinformática

Fork académico de **Diogo Esteves**, desenvolvido no contexto do mestrado em Bioinformática da Universidade do Minho. Reúne implementações didáticas de algoritmos de aprendizagem automática, exercícios e testes em Python.

[**Código**](src/si/) · [Exercícios](exercises/) · [Notebooks de exemplo](scripts/) · [Testes](tests/unit_tests/)

## Experimentar

Clona este fork e instala o pacote num ambiente Python separado:

```sh
git clone https://github.com/dasesteves/si.git
cd si
python -m pip install -r requirements.txt
python -m pip install -e .
```

O exemplo abaixo foi verificado em **Python 3.12**, com dependências já disponíveis. Embora `setup.py` ainda declare Python ≥ 3.7, o código usa sintaxe que requer **Python 3.10 ou posterior**. A compatibilidade com todas essas versões e uma instalação de raiz não foram verificadas. As versões das dependências não estão fixadas.

### Classificar duas amostras

Este exemplo usa apenas dados sintéticos e não precisa de descarregar um dataset:

```python
import numpy as np
from si.data.dataset import Dataset
from si.models.knn_classifier import KNNClassifier

treino = Dataset(
    X=np.array([[0.0], [1.0], [9.0], [10.0]]),
    y=np.array([0, 0, 1, 1]),
)
amostras = Dataset(X=np.array([[0.2], [9.8]]))

modelo = KNNClassifier(k=1)
modelo.fit(treino)
print(modelo.predict(amostras).tolist())
# [0, 1]
```

## Explorar e testar

- [Dataset e operações sobre dados](src/si/data/).
- [Modelos](src/si/models/) e [redes neuronais](src/si/neural_networks/).
- [Exercícios](exercises/) e [índice dos notebooks](scripts/).
- [Testes unitários](tests/unit_tests/), executáveis a partir da raiz após a instalação:

```sh
python -m pytest tests/unit_tests
```

Os notebooks podem exigir Jupyter, ficheiros de dados e dependências adicionais. Consulta cada exemplo antes de o executar. O projeto serve a aprendizagem dos algoritmos; os testes unitários não validam a sua utilização num contexto clínico.

## Origem e créditos

Este repositório é um fork de [jcorreia11/si](https://github.com/jcorreia11/si). O guia da unidade curricular de 2024–2025 e as suas instruções originais estão preservados em [Guia original do curso](docs/guia-original.md).

O pacote original declara inspiração e adaptação de [vmspereira/si](https://github.com/vmspereira/si), [cruz-f/si](https://github.com/cruz-f/si) e [jcorreia11/si](https://github.com/jcorreia11/si). Os créditos do curso e o código de origem mantêm-se atribuídos.
