import uuid
import random
import json
import os
import pathlib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum

# --- LLM Integration ---
try:
    from dotenv import load_dotenv
    from langchain_openai import AzureChatOpenAI
    
    # Load environment variables
    load_dotenv()
    
    # Azure OpenAI configuration
    DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
    ENDPOINT_URL = os.getenv("ENDPOINT_URL")
    AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
    
    # Initialize the LLM
    llm = AzureChatOpenAI(
        deployment_name=DEPLOYMENT_NAME,
        model_name="gpt-4",
        azure_endpoint=ENDPOINT_URL,
        api_key=AZURE_OPENAI_KEY,
        api_version="2023-05-15"
    )
    
    LLM_AVAILABLE = True
    print("Azure OpenAI LLM initialized successfully.")
except ImportError:
    LLM_AVAILABLE = False
    print("WARNING: Required packages for LLM integration not available. Install with: pip install python-dotenv langchain-openai")
except Exception as e:
    LLM_AVAILABLE = False
    print(f"WARNING: Failed to initialize Azure OpenAI LLM: {str(e)}")

# --- Enums ---

class CampaignStatus(str, Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    DELETED = "DELETED"
    # PENDING_REVIEW from previous version removed as per new schema

class Platform(str, Enum):
    GOOGLE = "GOOGLE"
    META = "META"
    LINKEDIN = "LINKEDIN"

# --- Constants (Adjusted for Daily Simulation) ---
MIN_DAILY_IMPRESSIONS = 500
MAX_DAILY_IMPRESSIONS = 20000
MIN_CTR = 0.01  # 1%
MAX_CTR = 0.08  # 8%
MIN_CPC = Decimal("0.30")
MAX_CPC = Decimal("3.50")
MIN_CONVERSION_RATE = 0.02 # 2%
MAX_CONVERSION_RATE = 0.15 # 15%
# Assuming a fixed conversion value for ROAS calculation for simplicity
ASSUMED_CONVERSION_VALUE = Decimal("50.00")
MIN_REACH_FACTOR = 0.6 # Min ratio of impressions that are unique reach
MAX_REACH_FACTOR = 0.9 # Max ratio
MIN_FREQUENCY = 1.1
MAX_FREQUENCY = 2.5


# --- Data Schemas (Pydantic Models based on new schema) ---

class CampaignInput(BaseModel):
    """Input schema for creating a new campaign."""
    platform_id: str = Field(..., example="G-Ads-12345")
    name: str = Field(..., example="Summer Sale 2025")
    status: CampaignStatus = Field(CampaignStatus.ACTIVE)
    objective: str = Field(..., example="CONVERSIONS")
    daily_budget: Decimal = Field(..., gt=Decimal(0), example=Decimal("500.00"))
    start_date: date
    end_date: Optional[date] = None
    platform: Platform

class CampaignUpdate(BaseModel):
    """Schema for updating an existing campaign. All fields are optional."""
    platform_id: Optional[str] = None
    name: Optional[str] = None
    status: Optional[CampaignStatus] = None
    objective: Optional[str] = None
    daily_budget: Optional[Decimal] = Field(None, gt=Decimal(0))
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    platform: Optional[Platform] = None

class Campaign(CampaignInput):
    """Represents a campaign stored in the database."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class CampaignMetric(BaseModel):
    """Represents metrics for a specific campaign on a specific day."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    campaign_id: uuid.UUID
    date: date
    impressions: int
    clicks: int
    ctr: Decimal = Field(..., description="Click-Through Rate")
    cpc: Decimal = Field(..., description="Cost Per Click")
    spend: Decimal
    conversions: int
    conversion_rate: Decimal
    roas: Decimal = Field(..., description="Return On Ad Spend")
    reach: int
    frequency: Decimal
    created_at: datetime = Field(default_factory=datetime.utcnow)

# --- Models for Insights and Recommendations ---

class CampaignInsight(BaseModel):
    """Represents an insight about a campaign's performance."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    campaign_id: uuid.UUID
    type: str = Field(..., example="performance", description="Type of insight like 'performance', 'audience', 'budget', etc.")
    description: str = Field(..., example="CTR has been declining over the past 3 days")
    severity: str = Field(..., example="medium", description="Impact level: 'low', 'medium', 'high'")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class CampaignRecommendation(BaseModel):
    """Represents a recommended action for a campaign."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    campaign_id: uuid.UUID
    insight_id: uuid.UUID
    action_type: str = Field(..., example="status_change", description="Type of action like 'status_change', 'budget_increase', etc.")
    description: str = Field(..., example="Pause the campaign until CTR improves")
    update_data: Dict[str, Any] = Field({}, example={"status": "PAUSED"}, description="The actual data to update in the campaign if applied")
    applied: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    applied_at: Optional[datetime] = None

class CampaignAnalysis(BaseModel):
    """Combined result of campaign analysis."""
    campaign_id: uuid.UUID
    insights: List[CampaignInsight]
    recommendations: List[CampaignRecommendation]
    summary: str

class RecommendationResponse(BaseModel):
    """Response after recommendation application."""
    success: bool
    message: str
    updated_campaign: Optional[Campaign] = None
    recommendation_id: uuid.UUID

# --- Database Persistence ---
# Define the data directory
DATA_DIR = pathlib.Path("./data")

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# File paths for each data type
CAMPAIGNS_FILE = DATA_DIR / "campaigns.json"
METRICS_FILE = DATA_DIR / "metrics.json"
INSIGHTS_FILE = DATA_DIR / "insights.json"
RECOMMENDATIONS_FILE = DATA_DIR / "recommendations.json"

# Custom JSON encoder to handle UUID, Decimal, dates, etc.
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

# Functions to convert data to/from JSON-compatible format
def object_to_dict(obj):
    """Convert Pydantic models and other objects to dictionaries for JSON serialization"""
    if hasattr(obj, 'model_dump'):
        # Handle Pydantic v2 models
        try:
            return {k: object_to_dict(v) for k, v in obj.model_dump().items()}
        except Exception as e:
            print(f"Error in model_dump conversion: {str(e)}")
            return str(obj)
    elif isinstance(obj, (datetime, date, Decimal, uuid.UUID)):
        # Convert special types to strings
        return str(obj)
    elif isinstance(obj, list):
        # Handle lists recursively
        return [object_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        # Handle dictionaries recursively
        return {k: object_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, Enum):
        return obj.value
    else:
        # Return basic types as is
        return obj

# Functions to save data to JSON files
def save_campaigns():
    """Save campaigns to JSON file"""
    campaigns_dict = {}
    for key, campaign in mock_campaign_db.items():
        campaigns_dict[str(key)] = object_to_dict(campaign)
    
    with open(CAMPAIGNS_FILE, 'w') as f:
        json.dump(campaigns_dict, f, cls=CustomJSONEncoder, indent=2)
    
    print(f"Saved {len(campaigns_dict)} campaigns to {CAMPAIGNS_FILE}")

def save_metrics():
    """Save metrics to JSON file"""
    metrics_dict = {}
    for key, metrics_list in mock_metrics_db.items():
        metrics_dict[str(key)] = object_to_dict(metrics_list)
    
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics_dict, f, cls=CustomJSONEncoder, indent=2)
    
    print(f"Saved metrics for {len(metrics_dict)} campaigns to {METRICS_FILE}")

def save_insights():
    """Save insights to JSON file"""
    insights_dict = {}
    for key, insights_list in mock_insights_db.items():
        insights_dict[str(key)] = object_to_dict(insights_list)
    
    with open(INSIGHTS_FILE, 'w') as f:
        json.dump(insights_dict, f, cls=CustomJSONEncoder, indent=2)
    
    print(f"Saved insights for {len(insights_dict)} campaigns to {INSIGHTS_FILE}")

def save_recommendations():
    """Save recommendations to JSON file"""
    recommendations_dict = {}
    for key, recommendations_list in mock_recommendations_db.items():
        recommendations_dict[str(key)] = object_to_dict(recommendations_list)
    
    with open(RECOMMENDATIONS_FILE, 'w') as f:
        json.dump(recommendations_dict, f, cls=CustomJSONEncoder, indent=2)
    
    print(f"Saved recommendations for {len(recommendations_dict)} campaigns to {RECOMMENDATIONS_FILE}")

def save_all_data():
    """Save all data to JSON files"""
    save_campaigns()
    save_metrics()
    save_insights()
    save_recommendations()
    print("All data saved successfully")

# Functions to load data from JSON files
def load_campaigns():
    """Load campaigns from JSON file"""
    global mock_campaign_db
    if not CAMPAIGNS_FILE.exists():
        print(f"No campaigns file found at {CAMPAIGNS_FILE}")
        return
    
    try:
        with open(CAMPAIGNS_FILE, 'r') as f:
            campaigns_dict = json.load(f)
        
        mock_campaign_db = {}
        for key, campaign_data in campaigns_dict.items():
            # Recreate Campaign object from dictionary
            campaign_input = {
                "platform_id": campaign_data["platform_id"],
                "name": campaign_data["name"],
                "status": CampaignStatus(campaign_data["status"]),
                "objective": campaign_data["objective"],
                "daily_budget": Decimal(campaign_data["daily_budget"]),
                "start_date": date.fromisoformat(campaign_data["start_date"]),
                "platform": Platform(campaign_data["platform"])
            }
            
            # Handle optional fields
            if campaign_data.get("end_date"):
                campaign_input["end_date"] = date.fromisoformat(campaign_data["end_date"])
            
            # Create Campaign object
            campaign = Campaign(**campaign_input)
            
            # Set ID and created_at
            campaign.id = uuid.UUID(campaign_data["id"])
            campaign.created_at = datetime.fromisoformat(campaign_data["created_at"])
            
            # Add to database
            mock_campaign_db[campaign.id] = campaign
        
        print(f"Loaded {len(mock_campaign_db)} campaigns from {CAMPAIGNS_FILE}")
    except Exception as e:
        print(f"Error loading campaigns: {str(e)}")

def load_metrics():
    """Load metrics from JSON file"""
    global mock_metrics_db
    if not METRICS_FILE.exists():
        print(f"No metrics file found at {METRICS_FILE}")
        return
    
    try:
        with open(METRICS_FILE, 'r') as f:
            metrics_dict = json.load(f)
        
        mock_metrics_db = {}
        for campaign_id_str, metrics_list_data in metrics_dict.items():
            campaign_id = uuid.UUID(campaign_id_str)
            metrics_list = []
            
            for metric_data in metrics_list_data:
                metric = CampaignMetric(
                    id=uuid.UUID(metric_data["id"]),
                    campaign_id=uuid.UUID(metric_data["campaign_id"]),
                    date=date.fromisoformat(metric_data["date"]),
                    impressions=int(metric_data["impressions"]),
                    clicks=int(metric_data["clicks"]),
                    ctr=Decimal(metric_data["ctr"]),
                    cpc=Decimal(metric_data["cpc"]),
                    spend=Decimal(metric_data["spend"]),
                    conversions=int(metric_data["conversions"]),
                    conversion_rate=Decimal(metric_data["conversion_rate"]),
                    roas=Decimal(metric_data["roas"]),
                    reach=int(metric_data["reach"]),
                    frequency=Decimal(metric_data["frequency"]),
                    created_at=datetime.fromisoformat(metric_data["created_at"])
                )
                metrics_list.append(metric)
            
            mock_metrics_db[campaign_id] = metrics_list
        
        print(f"Loaded metrics for {len(mock_metrics_db)} campaigns from {METRICS_FILE}")
    except Exception as e:
        print(f"Error loading metrics: {str(e)}")

def load_insights():
    """Load insights from JSON file"""
    global mock_insights_db
    if not INSIGHTS_FILE.exists():
        print(f"No insights file found at {INSIGHTS_FILE}")
        return
    
    try:
        with open(INSIGHTS_FILE, 'r') as f:
            insights_dict = json.load(f)
        
        mock_insights_db = {}
        for campaign_id_str, insights_list_data in insights_dict.items():
            campaign_id = uuid.UUID(campaign_id_str)
            insights_list = []
            
            for insight_data in insights_list_data:
                insight = CampaignInsight(
                    id=uuid.UUID(insight_data["id"]),
                    campaign_id=uuid.UUID(insight_data["campaign_id"]),
                    type=insight_data["type"],
                    description=insight_data["description"],
                    severity=insight_data["severity"],
                    created_at=datetime.fromisoformat(insight_data["created_at"])
                )
                insights_list.append(insight)
            
            mock_insights_db[campaign_id] = insights_list
        
        print(f"Loaded insights for {len(mock_insights_db)} campaigns from {INSIGHTS_FILE}")
    except Exception as e:
        print(f"Error loading insights: {str(e)}")

def load_recommendations():
    """Load recommendations from JSON file"""
    global mock_recommendations_db
    if not RECOMMENDATIONS_FILE.exists():
        print(f"No recommendations file found at {RECOMMENDATIONS_FILE}")
        return
    
    try:
        with open(RECOMMENDATIONS_FILE, 'r') as f:
            recommendations_dict = json.load(f)
        
        mock_recommendations_db = {}
        for campaign_id_str, recommendations_list_data in recommendations_dict.items():
            campaign_id = uuid.UUID(campaign_id_str)
            recommendations_list = []
            
            for rec_data in recommendations_list_data:
                recommendation = CampaignRecommendation(
                    id=uuid.UUID(rec_data["id"]),
                    campaign_id=uuid.UUID(rec_data["campaign_id"]),
                    insight_id=uuid.UUID(rec_data["insight_id"]),
                    action_type=rec_data["action_type"],
                    description=rec_data["description"],
                    update_data=rec_data["update_data"],
                    applied=rec_data["applied"],
                    created_at=datetime.fromisoformat(rec_data["created_at"])
                )
                
                # Handle optional fields
                if rec_data.get("applied_at"):
                    recommendation.applied_at = datetime.fromisoformat(rec_data["applied_at"])
                
                recommendations_list.append(recommendation)
            
            mock_recommendations_db[campaign_id] = recommendations_list
        
        print(f"Loaded recommendations for {len(mock_recommendations_db)} campaigns from {RECOMMENDATIONS_FILE}")
    except Exception as e:
        print(f"Error loading recommendations: {str(e)}")

def load_all_data():
    """Load all data from JSON files"""
    load_campaigns()
    load_metrics()
    load_insights()
    load_recommendations()
    print("All data loaded successfully")

# --- Mock Databases ---
# Mock storage for Campaigns
mock_campaign_db: Dict[uuid.UUID, Campaign] = {}

# Mock storage for daily Campaign Metrics (List per campaign_id)
mock_metrics_db: Dict[uuid.UUID, List[CampaignMetric]] = {}

# --- New Mock Databases ---
mock_insights_db: Dict[uuid.UUID, List[CampaignInsight]] = {}
mock_recommendations_db: Dict[uuid.UUID, List[CampaignRecommendation]] = {}

# --- Helper Functions ---

def simulate_daily_metrics(campaign_id: uuid.UUID, budget: Decimal, target_date: date, days_running: int) -> CampaignMetric:
    """Generates a simulated CampaignMetric object for a given campaign and date, applying a simple trend."""

    # --- Basic Trend Adjustment --- #
    # Slight improvement over the first ~30 days, capping at ~15-20% boost
    # This is a very simple model!
    trend_multiplier = Decimal(1.0 + min(days_running * 0.005, 0.15))
    spend_efficiency_factor = Decimal(1.0 - min(days_running * 0.003, 0.10)) # Simulates CPC potentially rising slightly

    # Adjust base random ranges slightly based on trend
    adj_max_impressions = int(MAX_DAILY_IMPRESSIONS * trend_multiplier)
    adj_min_impressions = int(MIN_DAILY_IMPRESSIONS * trend_multiplier)
    adj_max_ctr = MAX_CTR * float(trend_multiplier)
    adj_min_ctr = MIN_CTR * float(trend_multiplier)
    adj_min_cpc = MIN_CPC * spend_efficiency_factor
    adj_max_cpc = MAX_CPC * spend_efficiency_factor
    adj_min_conv_rate = MIN_CONVERSION_RATE * float(trend_multiplier)
    adj_max_conv_rate = MAX_CONVERSION_RATE * float(trend_multiplier)

    # Ensure min isn't greater than max after adjustment
    adj_min_impressions = min(adj_min_impressions, adj_max_impressions -1) if adj_max_impressions > MIN_DAILY_IMPRESSIONS else MIN_DAILY_IMPRESSIONS
    adj_min_ctr = min(adj_min_ctr, adj_max_ctr * 0.9) if adj_max_ctr > MIN_CTR else MIN_CTR
    adj_min_cpc = min(adj_min_cpc, adj_max_cpc * Decimal("0.9")) if adj_max_cpc > MIN_CPC else MIN_CPC
    adj_min_conv_rate = min(adj_min_conv_rate, adj_max_conv_rate * 0.9) if adj_max_conv_rate > MIN_CONVERSION_RATE else MIN_CONVERSION_RATE

    # --- Core Simulation Logic (using adjusted ranges) --- #
    impressions = random.randint(adj_min_impressions, adj_max_impressions)
    ctr = Decimal(random.uniform(adj_min_ctr, adj_max_ctr)).quantize(Decimal("0.0001"))
    clicks = int(impressions * ctr)

    if clicks > 0:
        cpc = Decimal(random.uniform(float(adj_min_cpc), float(adj_max_cpc))).quantize(Decimal("0.01"))
        potential_spend = (Decimal(clicks) * cpc).quantize(Decimal("0.01"))
        spend = min(potential_spend, budget)
        if spend < potential_spend and cpc > 0:
            clicks = int(spend / cpc)
            ctr = (Decimal(clicks) / Decimal(impressions)).quantize(Decimal("0.0001")) if impressions else Decimal(0)
        elif cpc == 0: # Avoid division by zero if CPC is somehow zero
             spend = Decimal(0)
    else:
        cpc = Decimal(0)
        spend = Decimal(0)

    if clicks > 0:
        conversion_rate = Decimal(random.uniform(adj_min_conv_rate, adj_max_conv_rate)).quantize(Decimal("0.0001"))
        conversions = int(clicks * conversion_rate)
    else:
        conversion_rate = Decimal(0)
        conversions = 0

    if spend > 0:
        total_conversion_value = Decimal(conversions) * ASSUMED_CONVERSION_VALUE
        roas = (total_conversion_value / spend).quantize(Decimal("0.01"))
    else:
        roas = Decimal(0)

    # Simulate reach and frequency
    reach_factor = Decimal(random.uniform(MIN_REACH_FACTOR, MAX_REACH_FACTOR))
    reach = int(impressions * reach_factor)
    frequency = (Decimal(impressions) / Decimal(reach)).quantize(Decimal("0.1")) if reach > 0 else Decimal(0)

    return CampaignMetric(
        campaign_id=campaign_id,
        date=target_date,
        impressions=impressions,
        clicks=clicks,
        ctr=ctr,
        cpc=cpc,
        spend=spend,
        conversions=conversions,
        conversion_rate=conversion_rate,
        roas=roas,
        reach=reach,
        frequency=frequency,
        created_at=datetime.utcnow()
    )

def seed_database():
    """Populates the mock databases with initial seed data."""
    print("Seeding database...")
    campaign_data = {
        "platform_id": "GA-Summer25-1",
        "name": "Summer Sale 2025",
        "status": CampaignStatus.ACTIVE,
        "objective": "SALES",
        "daily_budget": Decimal("500.00"),
        "start_date": date(2025, 4, 20),
        "platform": Platform.GOOGLE
    }
    # Use CampaignInput for validation before creating Campaign object
    validated_input = CampaignInput(**campaign_data)
    new_campaign = Campaign(**validated_input.model_dump())

    mock_campaign_db[new_campaign.id] = new_campaign
    mock_metrics_db[new_campaign.id] = []

    # Generate 7 days of metrics data
    for i in range(7):
        current_date = new_campaign.start_date + timedelta(days=i)
        daily_metric = simulate_daily_metrics(
            campaign_id=new_campaign.id,
            budget=new_campaign.daily_budget,
            target_date=current_date,
            days_running=i
        )
        mock_metrics_db[new_campaign.id].append(daily_metric)

    print(f"Seeded campaign {new_campaign.name} ({new_campaign.id}) with 7 days of metrics.")
    print("Database seeding complete.")

# --- Daily Population Logic ---

def populate_daily_metrics(target_date: date):
    """
    Iterates through active campaigns and populates metrics for the target_date
    if they don't already exist and the campaign is running.
    """
    print(f"--- Running daily metric population for {target_date} ---")
    populated_count = 0
    skipped_count = 0

    for campaign_id, campaign in mock_campaign_db.items():
        # 1. Check Status
        if campaign.status != CampaignStatus.ACTIVE:
            # print(f"Campaign {campaign.name} ({campaign_id}) is not ACTIVE. Skipping.")
            skipped_count += 1
            continue

        # 2. Check Date Range
        if target_date < campaign.start_date:
            # print(f"Campaign {campaign.name} ({campaign_id}) hasn't started yet ({campaign.start_date}). Skipping for {target_date}.")
            skipped_count += 1
            continue
        if campaign.end_date is not None and target_date > campaign.end_date:
            # print(f"Campaign {campaign.name} ({campaign_id}) ended on {campaign.end_date}. Skipping for {target_date}.")
            skipped_count += 1
            continue

        # Ensure metrics list exists for the campaign
        if campaign_id not in mock_metrics_db:
             mock_metrics_db[campaign_id] = []

        # 3. Check for Duplicates
        metrics_list = mock_metrics_db[campaign_id]
        date_exists = any(metric.date == target_date for metric in metrics_list)
        if date_exists:
            # print(f"Metrics for campaign {campaign.name} ({campaign_id}) on {target_date} already exist. Skipping.")
            skipped_count += 1
            continue

        # 4. Calculate days running & Simulate
        days_running = (target_date - campaign.start_date).days
        print(f"Populating metrics for campaign {campaign.name} ({campaign_id}) on {target_date} (Day {days_running})...")
        new_metric = simulate_daily_metrics(
            campaign_id=campaign_id,
            budget=campaign.daily_budget,
            target_date=target_date,
            days_running=days_running
        )

        # 5. Add to DB
        mock_metrics_db[campaign_id].append(new_metric)
        populated_count += 1

    print(f"--- Population complete for {target_date}. Populated: {populated_count}, Skipped: {skipped_count} ---")
    return {"populated": populated_count, "skipped": skipped_count, "target_date": target_date}

# --- FastAPI Application ---
app = FastAPI(
    title="Ad Platform API Simulator",
    description="Simulates managing campaigns and viewing daily performance metrics.",
    version="0.2.0"
)

@app.on_event("startup")
async def startup_event():
    """
    Initialize the application on startup.
    First tries to load data from JSON files, if no data exists, seeds the database.
    """
    try:
        # Try to load data from files
        load_all_data()
        
        # If no campaigns were loaded, seed the database
        if not mock_campaign_db:
            print("No existing data found, seeding database...")
            seed_database()
            # Save the seeded data
            save_all_data()
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        print("Falling back to seeding database...")
        seed_database()

@app.on_event("shutdown")
async def shutdown_event():
    """
    Save all data when the application shuts down.
    """
    try:
        save_all_data()
    except Exception as e:
        print(f"Error saving data during shutdown: {str(e)}")

# --- API Endpoints ---

@app.post("/campaigns/", response_model=Campaign, status_code=status.HTTP_201_CREATED)
async def create_campaign(campaign_input: CampaignInput):
    """
    Creates a new campaign according to the specified schema.
    Stores the campaign in the mock campaign DB and initializes its metrics list.
    """
    # Create the campaign object, ID and created_at are handled by Pydantic defaults
    new_campaign = Campaign(**campaign_input.model_dump())

    # Store in mock databases
    mock_campaign_db[new_campaign.id] = new_campaign
    mock_metrics_db[new_campaign.id] = [] # Initialize empty list for metrics
    
    # Save to persistent storage
    save_campaigns()
    save_metrics()

    return new_campaign

@app.get("/campaigns/{campaign_id}", response_model=Campaign)
async def get_campaign(campaign_id: uuid.UUID):
    """
    Retrieves details for a specific campaign by its UUID.
    """
    campaign = mock_campaign_db.get(campaign_id)
    if not campaign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign with ID {campaign_id} not found"
        )
    return campaign

@app.get("/campaigns/{campaign_id}/metrics/", response_model=List[CampaignMetric])
async def get_campaign_metrics(campaign_id: uuid.UUID):
    """
    Retrieves the list of daily metrics recorded for a specific campaign.
    """
    # First check if campaign exists
    if campaign_id not in mock_campaign_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign with ID {campaign_id} not found"
        )

    # Return metrics, defaults to empty list if somehow not initialized
    metrics = mock_metrics_db.get(campaign_id, [])
    return metrics

@app.get("/campaigns/", response_model=List[Campaign])
async def list_campaigns():
    """
    Lists all campaigns currently stored in the mock database.
    """
    return list(mock_campaign_db.values())

@app.post("/admin/populate-metrics/{date_str}", status_code=status.HTTP_200_OK)
async def trigger_populate_metrics(date_str: str):
    """
    Manually triggers the population of daily metrics for the given date.
    Date format should be YYYY-MM-DD.
    """
    try:
        target_date = date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Please use YYYY-MM-DD."
        )

    result = populate_daily_metrics(target_date)
    
    # Save the metrics to persistent storage
    save_metrics()
    
    return {"message": "Daily metrics population triggered.", "details": result}

@app.patch("/campaigns/{campaign_id}", response_model=Campaign)
async def update_campaign(campaign_id: uuid.UUID, update_data: CampaignUpdate):
    """
    Update an existing campaign with the provided data.
    Only the fields included in the request will be updated.
    """
    # Check if campaign exists
    if campaign_id not in mock_campaign_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign with ID {campaign_id} not found"
        )
    
    # Get existing campaign
    campaign = mock_campaign_db[campaign_id]
    
    # Convert update data to dict, excluding None values
    update_dict = {k: v for k, v in update_data.model_dump().items() if v is not None}
    
    # Special validation for start_date change if metrics already exist
    if 'start_date' in update_dict and campaign_id in mock_metrics_db and mock_metrics_db[campaign_id]:
        earliest_metric_date = min(metric.date for metric in mock_metrics_db[campaign_id])
        if update_dict['start_date'] > earliest_metric_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot change start_date to {update_dict['start_date']} as metrics exist from {earliest_metric_date}"
            )
    
    # Special validation for end_date
    if 'end_date' in update_dict:
        if update_dict['end_date'] is not None and update_dict['end_date'] < campaign.start_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"End date cannot be earlier than start date ({campaign.start_date})"
            )
    
    # If both start_date and end_date are being updated, check their relationship
    if 'start_date' in update_dict and 'end_date' in update_dict and update_dict['end_date'] is not None:
        if update_dict['start_date'] > update_dict['end_date']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Start date cannot be later than end date"
            )
    
    # Update campaign
    for key, value in update_dict.items():
        setattr(campaign, key, value)
    
    # Save the changes
    save_campaigns()
    
    # Return updated campaign
    return campaign

# --- Helper Functions for LLM ---

def analyze_campaign_metrics(campaign_id: uuid.UUID) -> CampaignAnalysis:
    """
    Analyze campaign metrics using LLM and generate insights and recommendations.
    Returns CampaignAnalysis object with insights, recommendations, and summary.
    """
    # Verify campaign exists
    if campaign_id not in mock_campaign_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign with ID {campaign_id} not found"
        )
    
    campaign = mock_campaign_db[campaign_id]
    metrics = mock_metrics_db.get(campaign_id, [])
    
    # If LLM is not available, provide basic analysis
    if not LLM_AVAILABLE or not metrics:
        # Create a basic insight and recommendation if metrics exist
        insights = []
        recommendations = []
        
        if metrics:
            try:
                # Sort metrics by date to analyze trends
                sorted_metrics = sorted(metrics, key=lambda x: x.date)
                latest_metrics = sorted_metrics[-1]
                
                # Calculate some basic trend indicators if we have multiple days of data
                if len(sorted_metrics) > 1:
                    try:
                        # Get metrics from previous day for comparison
                        prev_metrics = sorted_metrics[-2]
                        
                        # Calculate day-over-day changes
                        ctr_change = (latest_metrics.ctr - prev_metrics.ctr) / prev_metrics.ctr if prev_metrics.ctr else Decimal(0)
                        conversion_rate_change = (latest_metrics.conversion_rate - prev_metrics.conversion_rate) / prev_metrics.conversion_rate if prev_metrics.conversion_rate else Decimal(0)
                        spend_change = (latest_metrics.spend - prev_metrics.spend) / prev_metrics.spend if prev_metrics.spend else Decimal(0)
                        roas_change = (latest_metrics.roas - prev_metrics.roas) / prev_metrics.roas if prev_metrics.roas else Decimal(0)
                        
                        # Check if CTR is declining
                        if ctr_change < Decimal("-0.1"):  # 10% decrease
                            insight = CampaignInsight(
                                campaign_id=campaign_id,
                                type="performance",
                                description=f"CTR decreased by {abs(ctr_change)*100:.1f}% from {prev_metrics.date} to {latest_metrics.date}",
                                severity="medium" if ctr_change < Decimal("-0.2") else "low"
                            )
                            insights.append(insight)
                            
                            recommendation = CampaignRecommendation(
                                campaign_id=campaign_id,
                                insight_id=insight.id,
                                action_type="status_change",
                                description="Pause campaign due to declining CTR",
                                update_data={"status": CampaignStatus.PAUSED}
                            )
                            recommendations.append(recommendation)
                        
                        # Check if conversion rate is improving but budget is low
                        if conversion_rate_change > Decimal("0.1") and latest_metrics.roas > Decimal("2"):
                            insight = CampaignInsight(
                                campaign_id=campaign_id,
                                type="budget",
                                description=f"Conversion rate increased by {conversion_rate_change*100:.1f}% with strong ROAS of {latest_metrics.roas}",
                                severity="high"
                            )
                            insights.append(insight)
                            
                            # Calculate new budget (increase by 20%)
                            new_budget = campaign.daily_budget * Decimal("1.2")
                            recommendation = CampaignRecommendation(
                                campaign_id=campaign_id,
                                insight_id=insight.id,
                                action_type="budget_increase",
                                description=f"Increase daily budget from {campaign.daily_budget} to {new_budget} to capture more conversions",
                                update_data={"daily_budget": new_budget}
                            )
                            recommendations.append(recommendation)
                    except Exception as e:
                        print(f"Error analyzing metric trends: {str(e)}")
                
                # Single day analysis (no trends available)
                try:
                    # Check for low metrics that require immediate action
                    if latest_metrics.ctr < Decimal("0.01"):
                        insight = CampaignInsight(
                            campaign_id=campaign_id,
                            type="performance",
                            description=f"Very low CTR of {latest_metrics.ctr} on {latest_metrics.date}",
                            severity="high"
                        )
                        insights.append(insight)
                        
                        recommendation = CampaignRecommendation(
                            campaign_id=campaign_id,
                            insight_id=insight.id,
                            action_type="status_change",
                            description="Pause campaign due to extremely low CTR",
                            update_data={"status": CampaignStatus.PAUSED}
                        )
                        recommendations.append(recommendation)
                    
                    # Check for high spend with low conversions
                    if latest_metrics.spend > campaign.daily_budget * Decimal("0.7") and latest_metrics.conversions < 2:
                        insight = CampaignInsight(
                            campaign_id=campaign_id,
                            type="roi",
                            description=f"High spend ({latest_metrics.spend}) with low conversions ({latest_metrics.conversions})",
                            severity="high"
                        )
                        insights.append(insight)
                        
                        new_budget = max(campaign.daily_budget * Decimal("0.7"), Decimal("10"))
                        recommendation = CampaignRecommendation(
                            campaign_id=campaign_id,
                            insight_id=insight.id,
                            action_type="budget_decrease",
                            description=f"Decrease daily budget from {campaign.daily_budget} to {new_budget}",
                            update_data={"daily_budget": new_budget}
                        )
                        recommendations.append(recommendation)
                    
                    # Check for excellent performance
                    if latest_metrics.roas > Decimal("4") and latest_metrics.conversions > 5:
                        insight = CampaignInsight(
                            campaign_id=campaign_id,
                            type="performance",
                            description=f"Excellent ROAS of {latest_metrics.roas} with {latest_metrics.conversions} conversions",
                            severity="medium"
                        )
                        insights.append(insight)
                        
                        new_budget = campaign.daily_budget * Decimal("1.5")
                        recommendation = CampaignRecommendation(
                            campaign_id=campaign_id,
                            insight_id=insight.id,
                            action_type="budget_increase",
                            description=f"Increase daily budget significantly from {campaign.daily_budget} to {new_budget} to maximize results",
                            update_data={"daily_budget": new_budget}
                        )
                        recommendations.append(recommendation)
                except Exception as e:
                    print(f"Error in single-day metric analysis: {str(e)}")
                
                summary = f"Generated {len(insights)} insights and {len(recommendations)} recommendations based on {len(metrics)} days of data."
            except IndexError as e:
                print(f"Error accessing metrics data: {str(e)}")
                summary = "Error analyzing metrics: Unable to access required metrics data."
            except Exception as e:
                print(f"Unexpected error in fallback analysis: {str(e)}")
                summary = f"Error during fallback analysis: {str(e)}"
        else:
            summary = "No metrics available for analysis."
        
        analysis = CampaignAnalysis(
            campaign_id=campaign_id,
            insights=insights,
            recommendations=recommendations,
            summary=summary
        )
        
        # Store insights and recommendations
        mock_insights_db[campaign_id] = insights
        mock_recommendations_db[campaign_id] = recommendations
        
        # Save insights and recommendations to persistent storage
        save_insights()
        save_recommendations()
        
        return analysis
    
    # If LLM is available, use it for analysis
    try:
        # Prepare the campaign and metrics data for the LLM
        # Use a custom function to safely convert models to dictionaries
        def convert_to_dict(obj):
            if hasattr(obj, 'model_dump'):
                # Handle Pydantic v2 models
                try:
                    return {k: convert_to_dict(v) for k, v in obj.model_dump().items()}
                except Exception as e:
                    print(f"Error in model_dump conversion: {str(e)}")
                    # Fallback for objects with model_dump that might fail
                    return str(obj)
            elif hasattr(obj, 'dict'):
                # Handle Pydantic v1 models (fallback)
                try:
                    return {k: convert_to_dict(v) for k, v in obj.dict().items()}
                except Exception as e:
                    print(f"Error in dict conversion: {str(e)}")
                    return str(obj)
            elif isinstance(obj, (datetime, date, Decimal, uuid.UUID)):
                # Convert special types to strings
                return str(obj)
            elif isinstance(obj, list):
                # Handle lists recursively
                return [convert_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                # Handle dictionaries recursively
                return {k: convert_to_dict(v) for k, v in obj.items()}
            # Handle FieldInfo objects (common source of errors)
            elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'FieldInfo':
                return str(obj)
            # Handle Enum values
            elif isinstance(obj, Enum):
                return obj.value
            else:
                # Return basic types as is
                return obj
        
        # Safely prepare data for LLM analysis with better error handling
        try:
            # Convert campaign to a safe dictionary
            campaign_data = convert_to_dict(campaign)
            
            # Convert metrics to safe dictionaries
            metrics_data = [convert_to_dict(metric) for metric in metrics]
            
            # Check if conversion was successful
            if not isinstance(campaign_data, dict) or not all(isinstance(m, dict) for m in metrics_data):
                raise ValueError("Failed to convert campaign or metrics data to dictionaries")
                
            # Build prompt for the LLM
            prompt = f"""
            Analyze this campaign and its metrics. Provide insights and recommendations.
            
            Campaign details:
            {json.dumps(campaign_data, indent=2)}
            
            Metrics history:
            {json.dumps(metrics_data, indent=2)}
            
            Provide your analysis in this JSON format:
            {{
                "insights": [
                    {{
                        "type": "performance|audience|budget|roi|creative",
                        "description": "Detailed explanation of the insight",
                        "severity": "low|medium|high"
                    }}
                ],
                "recommendations": [
                    {{
                        "action_type": "status_change|budget_adjustment|audience_change",
                        "description": "Detailed explanation of the recommendation",
                        "update_data": {{
                            "field_to_update": "new_value"
                        }}
                    }}
                ],
                "summary": "A brief summary of the overall campaign performance and key recommendations."
            }}
            """
        except Exception as e:
            print(f"Error preparing data for LLM: {str(e)}")
            raise ValueError(f"Failed to prepare data for LLM analysis: {str(e)}")
            
        # Call the LLM with error handling
        try:
            response = llm.invoke(prompt)
        except Exception as e:
            print(f"Error during LLM invocation: {str(e)}")
            raise ValueError(f"LLM invocation failed: {str(e)}")
        
        # Extract and parse the JSON from the response
        try:
            # Look for JSON in the response, handling potential text preamble
            json_start = response.content.find('{')
            if json_start == -1:
                raise ValueError("No JSON found in LLM response")
                
            json_end = response.content.rfind('}') + 1
            json_str = response.content[json_start:json_end]
            
            try:
                analysis_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {str(e)}")
                print(f"JSON string: {json_str}")
                raise ValueError(f"Failed to parse JSON from LLM response: {str(e)}")
                
            # Validate required fields
            if not isinstance(analysis_data, dict):
                raise ValueError("LLM response is not a valid JSON object")
                
            if "insights" not in analysis_data or "recommendations" not in analysis_data:
                raise ValueError("LLM response missing required fields (insights or recommendations)")
                
            # Create insights from LLM response
            insights = []
            for insight_data in analysis_data.get('insights', []):
                insight = CampaignInsight(
                    campaign_id=campaign_id,
                    type=insight_data.get('type', 'unknown'),
                    description=insight_data.get('description', 'No description provided'),
                    severity=insight_data.get('severity', 'medium')
                )
                insights.append(insight)
            
            # Create recommendations from LLM response
            recommendations = []
            for rec_data in analysis_data.get('recommendations', []):
                # Map recommendation to actual update fields
                update_data = rec_data.get('update_data', {})
                action_type = rec_data.get('action_type', '').lower()
                
                # Handle different types of recommendations with specific logic
                if action_type == 'status_change':
                    # Convert string status to enum
                    if 'status' in update_data and isinstance(update_data['status'], str):
                        # Parse status value, handling variations
                        status_value = update_data['status'].upper()
                        if 'PAUS' in status_value:  # Handle "pause", "paused", etc.
                            update_data['status'] = CampaignStatus.PAUSED
                        elif 'ACTIV' in status_value or 'ENABLE' in status_value or 'RESUME' in status_value:
                            update_data['status'] = CampaignStatus.ACTIVE
                        elif 'DELET' in status_value or 'REMOV' in status_value:
                            update_data['status'] = CampaignStatus.DELETED
                        else:
                            # Default to current status if invalid
                            update_data['status'] = campaign.status
                
                elif action_type == 'budget_increase' or action_type == 'budget_adjustment' or action_type == 'budget_decrease':
                    # If no specific budget was provided, calculate based on recommendation type
                    if 'daily_budget' not in update_data:
                        if 'budget_increase' in action_type or 'increase' in rec_data.get('description', '').lower():
                            # Default increase by 20%
                            update_data['daily_budget'] = campaign.daily_budget * Decimal('1.2')
                        elif 'budget_decrease' in action_type or 'decrease' in rec_data.get('description', '').lower():
                            # Default decrease by 20%
                            update_data['daily_budget'] = campaign.daily_budget * Decimal('0.8')
                    elif isinstance(update_data.get('daily_budget'), str):
                        # Try to parse string budget value
                        try:
                            # Extract numeric part if string contains number
                            import re
                            numeric_part = re.search(r'(\d+(\.\d+)?)', update_data['daily_budget'])
                            if numeric_part:
                                update_data['daily_budget'] = Decimal(numeric_part.group(1))
                            else:
                                # If no number found, use percentage-based adjustment
                                if 'increase' in update_data['daily_budget'].lower():
                                    update_data['daily_budget'] = campaign.daily_budget * Decimal('1.2')
                                elif 'decrease' in update_data['daily_budget'].lower():
                                    update_data['daily_budget'] = campaign.daily_budget * Decimal('0.8')
                        except:
                            # Default to 10% increase if parsing fails
                            update_data['daily_budget'] = campaign.daily_budget * Decimal('1.1')
                    
                    # Ensure budget is never less than minimum threshold
                    if update_data.get('daily_budget', Decimal('0')) < Decimal('10'):
                        update_data['daily_budget'] = Decimal('10')
                
                # Find associated insight if possible
                insight_id = insights[0].id if insights else uuid.uuid4()
                
                recommendation = CampaignRecommendation(
                    campaign_id=campaign_id,
                    insight_id=insight_id,
                    action_type=action_type,
                    description=rec_data.get('description', 'No description provided'),
                    update_data=update_data
                )
                recommendations.append(recommendation)
            
            # Create final analysis object
            analysis = CampaignAnalysis(
                campaign_id=campaign_id,
                insights=insights,
                recommendations=recommendations,
                summary=analysis_data.get('summary', 'No summary provided by LLM.')
            )
            
            # Store insights and recommendations
            mock_insights_db[campaign_id] = insights
            mock_recommendations_db[campaign_id] = recommendations
            
            # Save insights and recommendations to persistent storage
            save_insights()
            save_recommendations()
            
            return analysis
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {e}")
            # Fall back to simple analysis
            return CampaignAnalysis(
                campaign_id=campaign_id,
                insights=[],
                recommendations=[],
                summary=f"Error analyzing campaign: {str(e)}"
            )
    
    except Exception as e:
        print(f"Error during LLM analysis: {e}")
        return CampaignAnalysis(
            campaign_id=campaign_id,
            insights=[],
            recommendations=[],
            summary=f"Error during analysis: {str(e)}"
        )

# --- Add new API Endpoints ---

@app.get("/campaigns/{campaign_id}/analyze", response_model=CampaignAnalysis)
async def get_campaign_insights(campaign_id: uuid.UUID):
    """
    Analyzes a campaign and its metrics to provide insights and recommendations.
    Uses LLM if available, otherwise falls back to rule-based analysis.
    """
    return analyze_campaign_metrics(campaign_id)

@app.post("/campaigns/{campaign_id}/recommendations/{recommendation_id}/apply", response_model=RecommendationResponse)
async def apply_recommendation(campaign_id: uuid.UUID, recommendation_id: uuid.UUID):
    """
    Applies a specific recommendation to a campaign.
    Updates campaign settings based on the recommendation's action_type and update_data.
    """
    # Verify campaign exists
    if campaign_id not in mock_campaign_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign with ID {campaign_id} not found"
        )
    
    # Get recommendations for this campaign
    recommendations = mock_recommendations_db.get(campaign_id, [])
    
    # Find the specific recommendation
    recommendation = next((r for r in recommendations if r.id == recommendation_id), None)
    if not recommendation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recommendation with ID {recommendation_id} not found"
        )
    
    # Check if already applied
    if recommendation.applied:
        return RecommendationResponse(
            success=False,
            message="Recommendation has already been applied.",
            recommendation_id=recommendation_id
        )
    
    try:
        # Get the campaign and prepare to update it
        campaign = mock_campaign_db[campaign_id]
        
        # Handle special cases based on action_type
        action_type = recommendation.action_type.lower()
        update_data = dict(recommendation.update_data)  # Create a copy to modify
        changes_made = []
        
        # Process update data based on action type
        if action_type == "status_change":
            # Ensure status is a valid enum value
            if 'status' in update_data:
                if isinstance(update_data['status'], str):
                    status_value = update_data['status'].upper()
                    if 'PAUS' in status_value:
                        update_data['status'] = CampaignStatus.PAUSED
                    elif any(term in status_value for term in ['ACTIV', 'ENABLE', 'RESUME', 'RUN']):
                        update_data['status'] = CampaignStatus.ACTIVE
                    elif any(term in status_value for term in ['DELET', 'REMOV', 'STOP']):
                        update_data['status'] = CampaignStatus.DELETED
                old_status = campaign.status
                campaign.status = update_data['status']
                changes_made.append(f"status: {old_status} -> {campaign.status}")
        
        elif action_type in ["budget_increase", "budget_decrease", "budget_adjustment"]:
            # Handle budget changes with proper validation
            if 'daily_budget' in update_data:
                # Convert to Decimal if string
                if isinstance(update_data['daily_budget'], str):
                    try:
                        # Extract numeric value if string contains a number
                        import re
                        numeric_match = re.search(r'(\d+(\.\d+)?)', update_data['daily_budget'])
                        if numeric_match:
                            new_budget = Decimal(numeric_match.group(1))
                        else:
                            # Apply percentage change if specified
                            if 'increase' in update_data['daily_budget'].lower():
                                # Extract percentage if available
                                pct_match = re.search(r'(\d+(\.\d+)?)%', update_data['daily_budget'])
                                pct = Decimal(pct_match.group(1))/100 if pct_match else Decimal('0.2')
                                new_budget = campaign.daily_budget * (Decimal('1') + pct)
                            elif 'decrease' in update_data['daily_budget'].lower():
                                pct_match = re.search(r'(\d+(\.\d+)?)%', update_data['daily_budget'])
                                pct = Decimal(pct_match.group(1))/100 if pct_match else Decimal('0.2')
                                new_budget = campaign.daily_budget * (Decimal('1') - pct)
                            else:
                                # Default to 20% change based on action type
                                modifier = Decimal('1.2') if 'increase' in action_type else Decimal('0.8')
                                new_budget = campaign.daily_budget * modifier
                    except Exception as e:
                        # Fallback to percentage-based adjustment
                        if 'increase' in action_type:
                            new_budget = campaign.daily_budget * Decimal('1.2')
                        elif 'decrease' in action_type:
                            new_budget = campaign.daily_budget * Decimal('0.8')
                        else:
                            new_budget = campaign.daily_budget * Decimal('1.1')
                else:
                    new_budget = Decimal(str(update_data['daily_budget']))
                
                # Apply minimum and maximum constraints
                min_budget = Decimal('10')
                max_budget = campaign.daily_budget * Decimal('5')  # Limit to 5x current budget
                new_budget = max(min_budget, min(new_budget, max_budget))
                
                old_budget = campaign.daily_budget
                campaign.daily_budget = new_budget
                changes_made.append(f"daily_budget: {old_budget} -> {new_budget}")
                
                # Calculate and log percentage change
                pct_change = ((new_budget / old_budget) - Decimal('1')) * Decimal('100')
                if abs(pct_change) >= Decimal('0.1'):  # Only log if change is at least 0.1%
                    print(f"Budget {pct_change > 0 and 'increased' or 'decreased'} by {abs(pct_change):.1f}%")
        
        # Handle any other fields that may need updating
        for key, value in update_data.items():
            # Skip keys we've already processed
            if key in ('status', 'daily_budget') and action_type in ('status_change', 'budget_increase', 'budget_decrease', 'budget_adjustment'):
                continue
                
            # Apply remaining updates if field exists on campaign
            if hasattr(campaign, key) and getattr(campaign, key) != value:
                old_value = getattr(campaign, key)
                setattr(campaign, key, value)
                changes_made.append(f"{key}: {old_value} -> {value}")
        
        # Mark recommendation as applied
        recommendation.applied = True
        recommendation.applied_at = datetime.utcnow()
        
        # Save the changes
        save_campaigns()
        save_recommendations()
        
        # Build result message
        if changes_made:
            result_message = f"Successfully applied recommendation: {recommendation.description}\nChanges: {', '.join(changes_made)}"
        else:
            result_message = f"Recommendation applied but no changes were needed: {recommendation.description}"
        
        return RecommendationResponse(
            success=True,
            message=result_message,
            updated_campaign=campaign,
            recommendation_id=recommendation_id
        )
    except Exception as e:
        return RecommendationResponse(
            success=False,
            message=f"Error applying recommendation: {str(e)}",
            recommendation_id=recommendation_id
        )

# Add a new admin route for manual DB operations
@app.post("/admin/db/save", status_code=status.HTTP_200_OK)
async def admin_save_database():
    """
    Manually trigger saving all data to JSON files.
    """
    try:
        save_all_data()
        return {"message": "Database successfully saved to disk."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving database: {str(e)}"
        )

@app.post("/admin/db/load", status_code=status.HTTP_200_OK)
async def admin_load_database():
    """
    Manually trigger loading all data from JSON files.
    """
    try:
        load_all_data()
        return {"message": "Database successfully loaded from disk."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading database: {str(e)}"
        )

# Replace the main method with improved port handling
if __name__ == "__main__":
    import uvicorn
    import socket
    
    # Default port
    port = 8000
    
    # Try alternative ports if default is in use
    for attempt in range(3):
        try:
            # Test if port is available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
            # If we get here, port is available
            break
        except OSError:
            print(f"Port {port} is in use, trying port {port + 1}")
            port += 1
    
    print(f"Starting server on port {port}")
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
