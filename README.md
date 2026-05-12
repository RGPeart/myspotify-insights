# MySpotify Insights 🎵

A data engineering portfolio project featuring an end-to-end ETL pipeline and ML-powered music recommendation engine using Spotify data and Azure cloud services.

![Architecture Diagram](docs/architecture_diagram.png)
*Coming soon*

## 🎯 Project Overview

MySpotify Insights demonstrates production-grade data engineering practices by building a complete music recommendation system that:
- Ingests personal Spotify listening history via API
- Processes data through a multi-stage ETL pipeline (Bronze → Silver → Gold)
- Trains a hybrid recommendation model (collaborative + content-based filtering)
- Serves recommendations via a REST API
- Visualizes insights through an interactive dashboard

**Tech Stack:** Python | Azure | FastAPI | Scikit-learn | Streamlit | Airflow

## 🚀 Features

- **Automated Data Pipeline**: Scheduled ingestion of Spotify data with error handling and data quality checks
- **ML Recommendations**: Personalized track suggestions based on listening patterns and audio features
- **REST API**: FastAPI service with OpenAPI documentation
- **Analytics Dashboard**: Interactive visualizations of listening habits and recommendation performance
- **Cloud Infrastructure**: Deployed on Azure with cost-optimized architecture

## 📊 Architecture
Spotify API → Azure Functions → Blob Storage (Bronze)
↓
ETL Pipeline (Airflow)
↓
Transformed Data (Silver/Gold)
↓
ML Model Training & Serving (FastAPI)
↓
Dashboard (Streamlit)

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.9+
- Azure account (free tier)
- Spotify Developer account
- Git

### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/YOUR_USERNAME/myspotify-insights.git
   cd myspotify-insights
```

2. **Create virtual environment**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
   cp .env.example .env
   # Edit .env with your Spotify and Azure credentials
```

5. **Run the ingestion pipeline**
```bash
   python src/ingestion/spotify_client.py
```

*Detailed setup instructions in [docs/setup.md](docs/setup.md)*

## 📖 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Documentation](docs/api_docs.md)
- [Setup Guide](docs/setup.md)

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/
```

## 📈 Roadmap

- [x] Project setup and PRD
- [ ] Spotify API integration
- [ ] ETL pipeline (Bronze → Silver → Gold)
- [ ] Recommendation model training
- [ ] FastAPI deployment
- [ ] Dashboard development
- [ ] CI/CD with GitHub Actions

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

## 👤 Author

**Ryan Peart**
- Portfolio: [Porfolio Link](https://rgpeart.github.io/portfolio)
- LinkedIn: [Ryan Peart](https://www.linkedin.com/in/ryan-peart/)
- GitHub: [@RGPeart](https://github.com/RGPeart)

## 🙏 Acknowledgments

- Spotify Web API for data access
- Azure for cloud infrastructure
- Open source libraries used in this project

---

⭐ Star this repo if you find it helpful!