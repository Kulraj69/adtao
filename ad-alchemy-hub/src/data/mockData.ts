
import { format, subDays } from "date-fns";

// Types for our data models
export interface Campaign {
  id: string;
  name: string;
  status: "active" | "paused" | "completed";
  budget: number;
  spend: number;
  startDate: string;
  endDate: string;
  ctr: number;
  cpc: number;
  conversions: number;
  platform: "google" | "facebook" | "instagram" | "tiktok";
}

export interface Creative {
  id: string;
  name: string;
  type: "image" | "video" | "text";
  campaignId: string;
  campaignName: string;
  impressions: number;
  clicks: number;
  ctr: number;
  createdDate: string;
  thumbnailUrl: string;
}

export interface Optimization {
  id: string;
  title: string;
  description: string;
  impact: string;
  campaignId?: string;
  campaignName?: string;
  date: string;
}

export interface DailyMetric {
  date: string;
  ctr: number;
  cpc: number;
  spend: number;
}

// Mock data generators
export const generateCampaigns = (): Campaign[] => {
  return [
    {
      id: "camp-1",
      name: "Summer Sale 2023",
      status: "active",
      budget: 5000,
      spend: 2340,
      startDate: "2023-06-01",
      endDate: "2023-08-15",
      ctr: 2.7,
      cpc: 0.85,
      conversions: 137,
      platform: "google",
    },
    {
      id: "camp-2",
      name: "New Product Launch",
      status: "active",
      budget: 8000,
      spend: 3200,
      startDate: "2023-05-15",
      endDate: "2023-07-30",
      ctr: 3.2,
      cpc: 0.72,
      conversions: 215,
      platform: "facebook",
    },
    {
      id: "camp-3",
      name: "Brand Awareness Q2",
      status: "completed",
      budget: 12000,
      spend: 12000,
      startDate: "2023-04-01",
      endDate: "2023-06-30",
      ctr: 1.9,
      cpc: 0.95,
      conversions: 310,
      platform: "instagram",
    },
    {
      id: "camp-4",
      name: "Holiday Promotion",
      status: "paused",
      budget: 6500,
      spend: 1200,
      startDate: "2023-11-01",
      endDate: "2023-12-31",
      ctr: 2.3,
      cpc: 0.79,
      conversions: 86,
      platform: "google",
    },
    {
      id: "camp-5",
      name: "Lead Generation Q3",
      status: "active",
      budget: 7500,
      spend: 4100,
      startDate: "2023-07-01",
      endDate: "2023-09-30",
      ctr: 1.8,
      cpc: 1.05,
      conversions: 175,
      platform: "facebook",
    },
  ];
};

export const generateCreatives = (): Creative[] => {
  return [
    {
      id: "creat-1",
      name: "Summer Sale Banner",
      type: "image",
      campaignId: "camp-1",
      campaignName: "Summer Sale 2023",
      impressions: 45000,
      clicks: 1215,
      ctr: 2.7,
      createdDate: "2023-05-28",
      thumbnailUrl: "https://placehold.co/300x200/4F46E5/FFFFFF/webp?text=Summer+Sale",
    },
    {
      id: "creat-2",
      name: "Product Demo Video",
      type: "video",
      campaignId: "camp-2",
      campaignName: "New Product Launch",
      impressions: 32000,
      clicks: 1024,
      ctr: 3.2,
      createdDate: "2023-05-10",
      thumbnailUrl: "https://placehold.co/300x200/10B981/FFFFFF/webp?text=Product+Demo",
    },
    {
      id: "creat-3",
      name: "Brand Story Ad",
      type: "image",
      campaignId: "camp-3",
      campaignName: "Brand Awareness Q2",
      impressions: 78000,
      clicks: 1482,
      ctr: 1.9,
      createdDate: "2023-04-05",
      thumbnailUrl: "https://placehold.co/300x200/F59E0B/FFFFFF/webp?text=Brand+Story",
    },
    {
      id: "creat-4",
      name: "Holiday Special Offer",
      type: "text",
      campaignId: "camp-4",
      campaignName: "Holiday Promotion",
      impressions: 12000,
      clicks: 276,
      ctr: 2.3,
      createdDate: "2023-10-25",
      thumbnailUrl: "https://placehold.co/300x200/EF4444/FFFFFF/webp?text=Holiday+Special",
    },
    {
      id: "creat-5",
      name: "Customer Testimonial",
      type: "video",
      campaignId: "camp-5",
      campaignName: "Lead Generation Q3",
      impressions: 25000,
      clicks: 450,
      ctr: 1.8,
      createdDate: "2023-06-28",
      thumbnailUrl: "https://placehold.co/300x200/6366F1/FFFFFF/webp?text=Testimonial",
    },
  ];
};

export const generateOptimizations = (): Optimization[] => {
  return [
    {
      id: "opt-1",
      title: "Increase bid for top performing keywords",
      description:
        "Your top 5 keywords are driving 70% of conversions but only using 30% of budget. We recommend increasing bids by 15%.",
      impact: "+22% conversions",
      campaignId: "camp-1",
      campaignName: "Summer Sale 2023",
      date: format(new Date(), "yyyy-MM-dd"),
    },
    {
      id: "opt-2",
      title: "Pause underperforming ad creatives",
      description:
        "3 ad creatives have CTR below 0.5% and high CPC. Pausing these could save budget for better performers.",
      impact: "-15% wasted spend",
      campaignId: "camp-2",
      campaignName: "New Product Launch",
      date: format(subDays(new Date(), 1), "yyyy-MM-dd"),
    },
    {
      id: "opt-3",
      title: "Optimize audience targeting",
      description:
        "Current audience segments show overlap. Consolidating to 4 segments could improve relevance and reduce ad fatigue.",
      impact: "+8% CTR",
      campaignId: "camp-5",
      campaignName: "Lead Generation Q3",
      date: format(subDays(new Date(), 2), "yyyy-MM-dd"),
    },
    {
      id: "opt-4",
      title: "Adjust ad schedule timing",
      description:
        "Performance data shows highest conversion rates between 6-9pm. Recommend shifting budget allocation to these hours.",
      impact: "+18% ROAS",
      campaignId: "camp-1",
      campaignName: "Summer Sale 2023",
      date: format(subDays(new Date(), 3), "yyyy-MM-dd"),
    },
  ];
};

export const generateDailyMetrics = (): DailyMetric[] => {
  const data: DailyMetric[] = [];
  const today = new Date();
  
  // Generate 30 days of data
  for (let i = 29; i >= 0; i--) {
    const date = subDays(today, i);
    const formattedDate = format(date, "MMM dd");
    
    // Generate some realistic looking data with small day-to-day variations
    const baseSpend = 500 + Math.random() * 300;
    const baseCtr = 2 + Math.random() * 2; // 2-4%
    const baseCpc = 0.7 + Math.random() * 0.6; // $0.7-$1.3
    
    // Add some weekly patterns and random variations
    const dayOfWeek = date.getDay();
    const weekendFactor = dayOfWeek === 0 || dayOfWeek === 6 ? 0.8 : 1;
    
    data.push({
      date: formattedDate,
      ctr: parseFloat((baseCtr * weekendFactor).toFixed(2)),
      cpc: parseFloat((baseCpc * (1/weekendFactor)).toFixed(2)),
      spend: parseFloat((baseSpend * weekendFactor).toFixed(2))
    });
  }
  
  return data;
};

// Summary metrics functions
export const getAccountMetrics = () => {
  const campaigns = generateCampaigns();
  const totalSpend = campaigns.reduce((sum, campaign) => sum + campaign.spend, 0);
  const weightedCtr = campaigns.reduce(
    (sum, campaign) => sum + campaign.ctr * campaign.spend,
    0
  ) / totalSpend;
  const weightedCpc = campaigns.reduce(
    (sum, campaign) => sum + campaign.cpc * campaign.spend,
    0
  ) / totalSpend;
  const totalConversions = campaigns.reduce(
    (sum, campaign) => sum + campaign.conversions,
    0
  );
  
  const conversionsPerDollar = totalConversions / totalSpend;
  const ROAS = 25 * conversionsPerDollar; // Assuming $25 average value per conversion
  
  return {
    totalSpend: totalSpend.toFixed(2),
    avgCtr: weightedCtr.toFixed(2),
    avgCpc: weightedCpc.toFixed(2),
    totalConversions,
    ROAS: ROAS.toFixed(2),
  };
};

