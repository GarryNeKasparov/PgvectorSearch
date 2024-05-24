CREATE VIEW products_view AS
SELECT
	uniq_id,
	description,
	selling_price,
	product_url,
	image,
	stars,
	word2vec_emb
FROM
	products
WITH CHECK OPTION;

CREATE OR REPLACE FUNCTION notify_generate_embeddings()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('generate_embeddings', NEW.description);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER generate_embeddings_trigger
INSTEAD OF INSERT ON products_view
FOR EACH ROW
EXECUTE FUNCTION notify_generate_embeddings();