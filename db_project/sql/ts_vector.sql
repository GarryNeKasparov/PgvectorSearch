ALTER TABLE products
ADD COLUMN ts_emb tsvector;

UPDATE products
SET ts_emb = to_tsvector(description);

CREATE INDEX products_gin 
ON products 
USING gin(ts_emb);

-- 49.8 min