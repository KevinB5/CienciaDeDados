# CienciaDeDados
 
Conclusões LAB1:
Outliers - caso não sejam valores disparatados, são valores importantes para especificações dos dados como a variância, correlação etc.
Z-score - é a standardização dos dados, pois faz simplesmente um shift dos dados para a origem, colocando a média a 0 e divide todos os pontos pelo desvio padrão, para ficar unitário. Este tipo de normalização mantém os outliers, sendo muito utilizado para casos que se tenha de lidar com PCA, clustering, logistic regression, SVM, neural networks, por estes dependerem de características dos dados, tais como a variância, correlação, etc. Vantagens: é uma normalização utilizada para facilitar por exemplo a regressão logistica, onde com os dados standardizados, mantendo a variância, se consegue reduzir os pesos da função logistica. O PCA é igualmente beneficiado pois precisa de manter as variâncias entre atributos para determinar quais as componentes principais.
Minmax Scaler - é a normalização que transforma os dados num intervalo entre 0 e 1 valores, onde os outliers perdem a sua importância, servindo mais para classificadores de distâncias ou decision trees, e em casos de processamento de imagem. Vantagens: esta normalização, como já foi dito anteriormente é bom para distâncias, ou seja, se tivermos atributos com escalas diferentes, esta normalização é a melhor para reduzir tempo de processamento e retirar um melhor proveito da vizinhança.

Conclusões LAB2:
Knn vs NaiveBayes - Depende da distribuição e concentração dos dados, o Knn é um classificador mais baseado em local features, e o NaiveBayes em global features. Numa base de dados como a IRIS, tendencialmente o Knn obtém melhores resultados, mas o NaiveBayes também pode ter resultados muito próximos. No caso de uma base de dados de um Hospital, em que os atributos são os vários sintomas de cada doente, o NaiveBayes porta-se melhor.
OverFitting em classificadores supervisionados - O método mais utilizado para compreender a evolução das accuracies ou erros do/s classificador/es, é o Cross Validation, uma técnica para separação dos dados em conjuntos de treino e teste, onde dependendo do número de observações escolhe-se o número de folds. Se as observações forem menos que 100 os Kfold tendem a igualar o número de observações, se forem mais que 100 tendem a variar entre os 10 e os 3 folds. Com esta técnica pode-se avaliar os resultados do classificador testando com um conjunto de dados de teste que o classificador nunca viu, tendo em conta a accuracy de cada k-fold. Tendo por exemplo dividido em 5 folds, convém saber o desvio padrão dos 5 para saber se a divisão das várias classes pelos folds está equilibrada. Para saber se o modelo está em overfitting convém testar as accuracies para o conjunto de treino e comparar com os de teste, não podendo haver muita diferença.
OverFitting em classificadores não supervisionados - Fazer n iterações variando os termos de regularização e ver a evolução das accuracies de teste e treino e avaliar.
 
DataSet DC:
A escolha da target variable passa por pre processar os dados, mas sem balance pois cada expert pode ter classificado mais resultados como bons e não maus, mas não invalida que esteja errado. Temos de ver a accuracy ou auc para escolher qual as melhores target_variables, e ver se ao concatenar experts temos um melhor consensus.

Pré-Processamento:
O pré-processamento é um passo importante no processo de mineração de texto. A frase "garbage in, garbage out" é particularmente aplicável a projetos de data mining e machine learning. Os métodos de coleta de dados geralmente são frouxamente controlados, resultando em valores de intervalo out-of-range (por exemplo, renda: −100),  combinações de dados impossíveis (por exemplo, sexo: masculino, grávidas: sim), missing values, etc. A análise de dados que não foram cuidadosamente selecionados para tais problemas pode produzir resultados enganosos. Assim, a representação e a qualidade dos dados são antes de tudo uma análise.

Normalização:
http://rajeshmahajan.com/standard-scaler-v-min-max-scaler-machine-learning/

PCA:
https://www.youtube.com/watch?v=FgakZw6K1QQ

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

O Pré-Processamento tem que ser feito durante o cross-validation.

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
