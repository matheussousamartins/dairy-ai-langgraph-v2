from app.db.connection import init_pools, close_pools
from app.rag.search import search_knowledge_base

init_pools()
try:
    r = search_knowledge_base(
        query='Quais são os cuidados gerais de laboratório segundo a IN 68?',
        table_name='embeddings_agente_4_qualidade_leite',
        search_type='hybrid_rrf',
        k=3,
    )
    print('ok', len(r))
    print(r[:1])
finally:
    close_pools()
