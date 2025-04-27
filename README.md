# Ad Tao

A FastAPI-powered Performance Marketing Agent designed for the HOLON × KBI Hackathon.

## Overview

Ad Tao lets users create mock Meta (Facebook/Instagram) ad campaigns, simulates realistic daily KPIs (CTR, CPC, ROAS, conversions), and runs an intelligence engine—rule checks plus optional GPT-4 augmentation—to generate plain-language insights and one-click optimization recommendations.

## Features

- **Campaign Management**: Create, read, update, and delete mock advertising campaigns
- **Metrics Simulation**: Daily generation of realistic performance metrics
- **AI-Powered Insights**: Automatic analysis of campaign performance with actionable recommendations
- **One-Click Optimization**: Apply recommendations with a single API call
- **Data Persistence**: All data stored in JSON files with automatic loading/saving

## Tech Stack

- **Backend**: FastAPI, Python
- **Intelligence Engine**: Rule-based analysis with optional Azure OpenAI GPT-4 integration
- **Data Storage**: Local JSON files
- **Frontend**: (To be implemented with Next.js)

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository
```
git clone https://github.com/yourusername/adtao.git
cd adtao
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. (Optional) Set up OpenAI integration
   - Create a `.env` file in the project root
   - Add the following values:
   ```
   DEPLOYMENT_NAME=your-deployment-name
   ENDPOINT_URL=your-azure-endpoint
   AZURE_OPENAI_KEY=your-azure-key
   ```

### Running the API

```
python backend/main.py
```

The API will start on http://127.0.0.1:8000 (or the next available port if 8000 is taken)

## API Endpoints

### Campaigns
- `POST /campaigns/` - Create a new campaign
- `GET /campaigns/` - List all campaigns
- `GET /campaigns/{campaign_id}` - Get a specific campaign
- `PATCH /campaigns/{campaign_id}` - Update a campaign

### Metrics
- `GET /campaigns/{campaign_id}/metrics/` - Get metrics for a campaign
- `POST /admin/populate-metrics/{date_str}` - Generate metrics for a specific date

### Analysis & Recommendations
- `GET /campaigns/{campaign_id}/analyze` - Get insights and recommendations
- `POST /campaigns/{campaign_id}/recommendations/{recommendation_id}/apply` - Apply a recommendation

### Administration
- `POST /admin/db/save` - Save all data to disk
- `POST /admin/db/load` - Load all data from disk

## Project Structure

- `backend/` - FastAPI backend code
- `data/` - JSON files for persistent storage
- (Future) `frontend/` - Next.js frontend

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built for the HOLON × KBI Hackathon
- Designed for small-to-mid-size advertisers and agency juniors 