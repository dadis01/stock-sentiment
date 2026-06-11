web: streamlit run web_app/app.py --server.port=$PORT --server.address=0.0.0.0
collector: python data_collector/collector.py
analyzer: uvicorn data_analyzer.analyzer:app --host=0.0.0.0 --port=${PORT:-8000}
