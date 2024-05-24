WITH emb AS (
	SELECT word2vec_emb
	FROM products LIMIT 1
)
SELECT category_name FROM products 
ORDER BY word2vec_emb <-> (SELECT word2vec_emb FROM emb)  ASC LIMIT 5