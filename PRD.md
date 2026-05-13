# Product Requirements Document: Music Recommendation Engine

**Project Name:** MySpotify Insights  
**Version:** 1.0  
**Author:** Ryan Peart  
**Last Updated:** May 12, 2026  
**Status:** Draft

---

## 1. Overview

MySpotify Insights is a data engineering portfolio project that demonstrates end-to-end pipeline development, cloud infrastructure management, and machine learning integration. The project builds a music recommendation system powered by Spotify data, featuring a robust ETL pipeline, intelligent recommendation engine, and interactive web analytics dashboard.

**Problem Statement:**  
Music streaming platforms generate massive amounts of user behavior and audio feature data, but understanding how to process, model, and serve recommendations at scale requires sophisticated data engineering. This project simulates real-world challenges faced by companies like Spotify, demonstrating skills in data ingestion, transformation, storage optimization, and ML model deployment.

**Solution:**  
A full-stack data product that:
- Ingests music data from Spotify API (tracks, artists, audio features)
- Processes and transforms raw data through an ETL pipeline
- Generates personalized music recommendations using collaborative filtering
- Exposes recommendations via a RESTful API
- Visualizes pipeline performance and recommendation quality through a dashboard

---

## 2. Goals and Non-Goals

### Goals
1. **Demonstrate Data Engineering Skills**
   - Build production-quality ETL pipeline with orchestration
   - Implement data quality checks and error handling
   - Design scalable data models and storage architecture

2. **Showcase Cloud Infrastructure Knowledge**
   - Utilize Azure services cost-effectively (Blob Storage, Functions, App Service)
   - Implement CI/CD practices with GitHub Actions
   - Deploy live, publicly accessible demo

3. **Integrate Machine Learning**
   - Build recommendation engine using audio features and user behavior patterns
   - Deploy ML model as a REST API
   - Track model performance metrics

4. **Create Portfolio-Ready Deliverable**
   - Comprehensive documentation with architecture diagrams
   - Clean, well-tested codebase
   - Live demo with compelling use case
   - Blog post explaining technical decisions

### Non-Goals
1. **Not building a production music streaming service** - Focus is on data engineering infrastructure, not full-featured product
2. **Not implementing real-time streaming** - Batch processing is sufficient for portfolio demonstration (can be Phase 2)
3. **Not optimizing for scale** - Designed for thousands of tracks, not millions (cost constraint)
4. **Not building mobile apps** - Web-only interface sufficient for demo

---

## 3. Target Users

### Primary Audience: Recruiters & Hiring Managers
**Profile:**
- Data Engineering, Analytics Engineering, or ML Engineering roles
- Companies: Spotify, WHOOP, Microsoft, or similar tech companies
- Looking for candidates with end-to-end pipeline development experience

**What they want to see:**
- Clean, documented code with best practices
- Understanding of cloud architecture and cost optimization
- Ability to design scalable data models
- Integration of multiple technologies (ETL, ML, APIs, cloud)
- Problem-solving and technical decision-making

### Secondary Audience: Technical Community
**Profile:**
- Engineers learning data pipelines
- Portfolio project inspiration seekers
- Blog readers interested in Spotify API + Azure

**What they want:**
- Step-by-step technical breakdown
- Open-source code to learn from
- Architectural insights and tradeoffs

---

## 4. Core Features

### Feature 1: Data Ingestion Pipeline
**Description:** Automated system to extract music data from Spotify API and load into Azure Blob Storage.

**Technical Details:**
- Azure Functions (timer-triggered) for scheduled API calls
- Rate limiting and error handling for API reliability
- Incremental data loading (avoid reprocessing)
- Store raw JSON data in "bronze" layer

**Data Sources:**
- Spotify Web API endpoints: tracks, artists, audio features, genres
- Sample user listening history (simulated or personal Spotify data)

---

### Feature 2: ETL Pipeline with Orchestration
**Description:** Transform raw Spotify data into analytics-ready models with data quality checks.

**Technical Details:**
- **Bronze → Silver:** Clean and normalize JSON data
- **Silver → Gold:** Create dimensional models (fact_listening_history, dim_tracks, dim_artists)
- Orchestration via Airflow (Docker) or Prefect
- Data quality tests: null checks, schema validation, duplicate detection
- Store processed data in Azure SQL Database (free tier) or Parquet files

**Transformations:**
- Audio feature normalization (scale 0-1)
- Genre categorization and tagging
- Popularity scoring algorithms
- Time-based listening pattern aggregations

---

### Feature 3: Recommendation Engine
**Description:** Machine learning model that generates personalized track recommendations.

**Technical Details:**
- **Algorithm:** Collaborative filtering (user-item matrix) + content-based filtering (audio features)
- **Training:** Scikit-learn or lightweight library
- **Features:** danceability, energy, valence, tempo, genre similarity, artist popularity
- **Output:** Top 10 recommended tracks per user with confidence scores
- Model retraining schedule (weekly batch job)

**Recommendation Logic:**
- Hybrid approach: 70% collaborative filtering, 30% content similarity
- Cold-start handling: Use audio features for new users/tracks
- Diversity bonus: Penalize genre over-concentration

---

### Feature 4: REST API for Recommendations
**Description:** FastAPI service that serves recommendations and track metadata.

**Endpoints:**
- `GET /recommendations/{user_id}` - Get top N recommendations
- `GET /tracks/{track_id}` - Get track details and audio features
- `GET /artists/{artist_id}` - Get artist information
- `GET /health` - Service health check

**Deployment:**
- Azure App Service (free tier) or Azure Container Instances
- OpenAPI documentation (Swagger UI)
- Response caching for performance

---

### Feature 5: Analytics Dashboard
**Description:** Web-based dashboard showing pipeline metrics and recommendation insights.

**Metrics Displayed:**
- Pipeline run status and data freshness
- Recommendation model performance (precision, recall, diversity)
- Data quality KPIs (completeness, validity)
- Most recommended tracks/artists
- Audio feature distributions

**Technology:**
- Streamlit or React + Chart.js
- Deployed on Streamlit Community Cloud or Azure Static Web Apps (free)

---

## 5. User Stories

### For Recruiters/Hiring Managers

**US-1:** As a hiring manager, I want to see a live demo URL so I can quickly evaluate the candidate's work without setup.

**US-2:** As a technical recruiter, I want to read a clear README with architecture diagrams so I can understand the system design at a glance.

**US-3:** As a data engineering manager, I want to review the codebase and see data quality checks so I can assess production-readiness thinking.

**US-4:** As an interviewer, I want to see metrics and monitoring so I can ask questions about operational considerations.

---

### For Data Pipeline (Technical)

**US-5:** As the ETL pipeline, I want to incrementally load only new Spotify data so I minimize API calls and costs.

**US-6:** As the data quality module, I want to validate schemas and detect anomalies so bad data doesn't corrupt downstream models.

**US-7:** As the orchestrator, I want to retry failed tasks with exponential backoff so transient errors don't break the pipeline.

---

### For End Users (Demo Context)

**US-8:** As a music listener, I want to receive 10 personalized track recommendations so I can discover new music matching my taste.

**US-9:** As a user, I want to see why a track was recommended (e.g., "Similar to artists you like") so recommendations feel transparent.

**US-10:** As a dashboard viewer, I want to explore audio feature distributions so I can understand what makes recommendations work.

---

## 6. Success Metrics

### Technical Metrics (Primary Focus)

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Pipeline Reliability** | 95%+ successful runs | Airflow task success rate |
| **Data Freshness** | Updated within 24 hours | Timestamp of latest ingestion |
| **API Response Time** | < 500ms (p95) | FastAPI logging/monitoring |
| **Data Quality Score** | 90%+ passing checks | Great Expectations or custom tests |
| **Code Coverage** | > 70% | pytest coverage reports |

### Model Performance Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Recommendation Precision@10** | > 0.20 | Offline evaluation on test set |
| **Genre Diversity** | > 3 genres in top 10 | Diversity calculation |
| **Cold Start Coverage** | > 80% of tracks | % tracks that can be recommended |

### Portfolio Impact Metrics (Qualitative)

- GitHub stars/forks received
- Blog post engagement (views, shares)
- Recruiter inquiries mentioning the project
- Interview conversations driven by project

---

## 7. Open Questions

### Technical Decisions
1. **Q:** Should we use Airflow (more popular) or Prefect (easier setup) for orchestration?  
   **Status:** Lean toward Airflow for resume value, but Prefect if setup becomes blocker

2. **Q:** Azure SQL Database (free tier) vs. DuckDB (file-based) for storage?  
   **Status:** Start with Parquet + DuckDB for cost, migrate to Azure SQL if needed

3. **Q:** How much historical data to collect before building recommendations?  
   **Status:** Target 1,000+ tracks and 10,000+ listening events (can simulate)

### Scope Questions
4. **Q:** Should we include a "feedback loop" where users rate recommendations?  
   **Status:** Nice-to-have for Phase 2, not MVP

5. **Q:** Do we need real user authentication or can demo use hard-coded sample users?  
   **Status:** Use sample users for MVP, document as future enhancement

6. **Q:** Should the dashboard be public or password-protected?  
   **Status:** Public demo is better for portfolio accessibility

### Data Questions
7. **Q:** Can we use personal Spotify listening history or need to simulate user data?  
   **Status:** Investigate Spotify API permissions, fallback to simulated data

8. **Q:** What's the minimum viable dataset size to show credible recommendations?  
   **Status:** 500 tracks x 5 users = 2,500 interactions minimum

---

## 8. Milestones

### Milestone 1: Foundation & Data Ingestion (Week 1)
**Deliverables:**
- [ ] GitHub repository set up with README
- [ ] Azure account configured with resource group
- [ ] Spotify API integration working
- [ ] Azure Function that fetches track data to Blob Storage
- [ ] Initial architecture diagram

**Success Criteria:** Can extract 100+ tracks with audio features and store as JSON in Azure

---

### Milestone 2: ETL Pipeline & Data Models (Week 2)
**Deliverables:**
- [ ] Bronze → Silver → Gold data transformations
- [ ] Dimensional data models (fact/dim tables)
- [ ] Airflow/Prefect DAG running locally
- [ ] Data quality tests implemented
- [ ] Parquet/SQL storage working

**Success Criteria:** Automated pipeline processes raw Spotify data into analytics-ready tables

---

### Milestone 3: Recommendation Engine & API (Week 3)
**Deliverables:**
- [ ] Trained collaborative filtering model
- [ ] Model evaluation metrics calculated
- [ ] FastAPI service with 4 core endpoints
- [ ] API deployed to Azure App Service
- [ ] OpenAPI documentation live

**Success Criteria:** API returns 10 recommendations per user with <500ms latency

---

### Milestone 4: Dashboard & Portfolio Polish (Week 4)
**Deliverables:**
- [ ] Interactive dashboard with 5+ visualizations
- [ ] Dashboard deployed publicly
- [ ] Architecture diagram finalized
- [ ] README with setup instructions and demo links
- [ ] Code cleaned, commented, tested
- [ ] Demo video or GIF created
- [ ] Blog post drafted

**Success Criteria:** Complete portfolio-ready project with public demo URL and comprehensive docs

---

### Milestone 5: Optional Enhancements (Post-MVP)
**Future Ideas:**
- Real-time streaming with Azure Event Hubs
- A/B testing framework for recommendation algorithms
- Integration with additional music APIs (Last.fm, MusicBrainz)
- User feedback loop and retraining pipeline
- Cost optimization analysis and report

---

## Appendix

### Tech Stack Summary
- **Cloud:** Azure (Blob Storage, Functions, App Service, SQL Database)
- **Orchestration:** Airflow or Prefect
- **Processing:** Python (pandas, polars)
- **ML:** Scikit-learn, Surprise (collaborative filtering)
- **API:** FastAPI
- **Dashboard:** Streamlit or React
- **CI/CD:** GitHub Actions
- **IaC:** Azure CLI or Terraform (optional)

### Estimated Costs
- Azure Free Tier: $0 (first 12 months)
- After free tier: ~$5-10/month (Blob Storage + App Service)
- Streamlit Community Cloud: $0

### Timeline
- **Total Duration:** 4 weeks (part-time)
- **Effort:** 10-15 hours/week
- **Target Completion:** June 9, 2026