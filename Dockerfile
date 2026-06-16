# Imagem base Python
FROM python:3.12-slim

# Diretório de trabalho dentro do container
WORKDIR /app

# Copiar dependências primeiro (otimiza o cache do Docker)
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o projeto inteiro
COPY . .

# Expor a porta do Streamlit
EXPOSE 8501

# Comando para rodar o dashboard
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]