import seaborn as sns
import matplotlib.pyplot as plt

# models = ['Word2Vec', 'LSTM', 'MiniLM']
# times = [4.8, 24.6, 15.5]
# fig = sns.barplot(x=models, y=times)
# plt.title('Сравнение скорости получения представлений текстов')
# plt.xlabel('Модель')
# plt.ylabel('Время (мин)')
# fig.get_figure().savefig('time.png')

# models = ['Word2Vec', 'LSTM', 'MiniLM']
# scores = [0.76, 0.74, 0.8]
# fig = sns.barplot(x=models, y=scores)
# plt.title('Сравнение качества представлений')
# plt.xlabel('Модель')
# plt.ylabel('MAP@5')
# fig.get_figure().savefig('scores.png')

# vec_size = [32, 384]
# times = [2.549, 15.02]
# fig = sns.barplot(x=vec_size, y=times)
# plt.title('Сравнение скорости поиска')
# plt.xlabel('Размер представлений')
# plt.ylabel('Время (сек)')
# fig.get_figure().savefig('speed.png')

# построение индекса HNSW 43.6 мин
# построение индекса GIN 1.2 мин
#  wv   lstm  tr
# [0.82 0.75 0.88]

search_type = ["Exact", "HNSW"]
scores = [0.76, 0.69]
fig = sns.barplot(x=search_type, y=scores)
plt.title("Сравнение качества точного и приближенного поиска")
plt.xlabel("Тип поиска")
plt.ylabel("MAP@5")
fig.get_figure().savefig("index.png")
