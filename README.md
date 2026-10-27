uv run inspect_dataset.py
uv run inspect_descriptions.py


what we could infer from inspect_descriptions is that some listings have the total price, while some only have the monthly payment which cannot be attributed as the full price so what we will do is create `price_aed` column and `monthly_payment_aed` column, and depending on if the total price is given we can eitheir leave `price_aed` empty and store the given information

mileage extraction looks promising as 61/100 records have mentions about mileage we can also extract other information such as transmission, fuel types, specification, warranty and down payment from the examples


uv run python -m src.preprocessing

uv run python -m src.retrieval
uv run python -m src.semantic_retrieval                                                                                               
uv run python -m tests.test_hybrid_retrieval 

main.py -> backend

gemini_client.py
chat_service.py


run the application - 
uv run uvicorn main:app --reload