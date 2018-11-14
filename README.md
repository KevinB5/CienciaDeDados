﻿# CienciaDeDados
 
DataSet DC:
A escolha da target variable passa por pre processar os dados, mas sem balance pois cada expert pode ter classificado mais resultados como bons e não maus, mas não invalida que esteja errado. Temos de ver a accuracy ou auc para escolher qual as melhores target_variables, e ver se ao concatenar experts temos um melhor consensus.

Pré-Processamento:
O pré-processamento é um passo importante no processo de mineração de texto. A frase "garbage in, garbage out" é particularmente aplicável a projetos de data mining e machine learning. Os métodos de coleta de dados geralmente são frouxamente controlados, resultando em valores de intervalo out-of-range (por exemplo, renda: −100),  combinações de dados impossíveis (por exemplo, sexo: masculino, grávidas: sim), missing values, etc. A análise de dados que não foram cuidadosamente selecionados para tais problemas pode produzir resultados enganosos. Assim, a representação e a qualidade dos dados são antes de tudo uma análise.

Razões para Pré-Processamento:
1. Missing headers in the csv file
2. Multiple variables are stored in one column
3. Column data contains inconsistent unit values
4. Missing values
5. An empty row in the data
6. Duplicate records in the data
7. Non-ASCII characters
8. Column headers are values and not variable names

Albert Einstein once said, "if you judge a fish on its ability to climb a tree, it will live its whole life believing that it is stupid." This quote really highlights the importance of choosing the right evaluation metric.

Para dados unbalanced é necessário ver métricas como AUC, recall, precision, pois da accuracy não se pode tirar conclusões.

Exploration:
Data exploration is an approach similar to initial data analysis, whereby a data analyst uses visual exploration to understand what is in a dataset and the characteristics of the data, rather than through traditional data management systems[1]. These characteristics can include size or amount of data, completeness of the data, correctness of the data, possible relationships amongst data elements or files/tables in the data.

Some more examples of parametric machine learning algorithms include:
-Logistic Regression
-Linear Discriminant Analysis
-Perceptron
-Naive Bayes
-Simple Neural Networks

Some more examples of popular nonparametric machine learning algorithms are:
-k-Nearest Neighbors
-Decision Trees like CART and C4.5
-Support Vector Machines

Exemplo:
https://github.com/rhiever/Data-Analysis-and-Machine-Learning-Projects/blob/master/example-data-science-notebook/Example%20Machine%20Learning%20Notebook.ipynb

Dealing with Imbalanced Data:
https://www.marcoaltini.com/blog/dealing-with-imbalanced-data-undersampling-oversampling-and-proper-cross-validation

Vídeos:
Visualizando uma Árvore de Decisão - Receitas de Aprendizado de Máquina #2: https://www.youtube.com/watch?v=tNa99PG8hR8
Decision Tree Algorithm | Decision Tree in Python | Machine Learning Algorithms | Edureka: https://youtu.be/qDcl-FRnwSU
Machine Learning 267 k Fold Cross Validation in Python: https://youtu.be/QGygjpBGG20
Data Science Tutorial | Data Science for Beginners | Data Science with Python Tutorial | Simplilearn: https://youtu.be/jNeUBWrrRsQ
