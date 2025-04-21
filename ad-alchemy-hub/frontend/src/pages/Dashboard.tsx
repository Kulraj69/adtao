
import React, { useState } from "react";
import StatsCard from "@/components/StatsCard";
import PerformanceChart from "@/components/PerformanceChart";
import OptimizationCard from "@/components/OptimizationCard";
import DataTable from "@/components/DataTable";
import { 
  generateDailyMetrics, 
  generateOptimizations, 
  generateCampaigns,
  getAccountMetrics
} from "@/data/mockData";
import { toast } from "sonner";

const Dashboard = () => {
  const [optimizations, setOptimizations] = useState(generateOptimizations());
  const metrics = getAccountMetrics();
  const performanceData = generateDailyMetrics();
  const campaigns = generateCampaigns();
  
  const recentCampaigns = campaigns
    .sort((a, b) => new Date(b.startDate).getTime() - new Date(a.startDate).getTime())
    .slice(0, 5);
  
  const handleApplyOptimization = (id: string) => {
    toast.success("Optimization applied successfully!");
    setOptimizations(optimizations.filter((opt) => opt.id !== id));
  };
  
  const handleIgnoreOptimization = (id: string) => {
    toast.info("Optimization ignored.");
    setOptimizations(optimizations.filter((opt) => opt.id !== id));
  };

  const campaignColumns = [
    {
      header: "Campaign",
      accessor: "name",
      className: "font-medium text-gray-900",
    },
    {
      header: "Status",
      accessor: (row: any) => (
        <span
          className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${
            row.status === "active"
              ? "bg-green-100 text-green-800"
              : row.status === "paused"
              ? "bg-yellow-100 text-yellow-800"
              : "bg-gray-100 text-gray-800"
          }`}
        >
          {row.status.charAt(0).toUpperCase() + row.status.slice(1)}
        </span>
      ),
    },
    { header: "CTR", accessor: (row: any) => `${row.ctr}%` },
    { header: "CPC", accessor: (row: any) => `$${row.cpc.toFixed(2)}` },
    { header: "Conversions", accessor: "conversions" },
    {
      header: "Platform",
      accessor: (row: any) => (
        <span className="capitalize">{row.platform}</span>
      ),
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-500">Welcome to your marketing performance overview.</p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Total Ad Spend"
          value={`$${metrics.totalSpend}`}
          trend={{ value: 12.5, isUpward: true }}
        />
        <StatsCard
          title="Average CTR"
          value={`${metrics.avgCtr}%`}
          trend={{ value: 8.3, isUpward: true }}
        />
        <StatsCard
          title="Average CPC"
          value={`$${metrics.avgCpc}`}
          trend={{ value: 3.2, isUpward: false }}
        />
        <StatsCard
          title="ROAS"
          value={`${metrics.ROAS}x`}
          trend={{ value: 5.7, isUpward: true }}
        />
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <PerformanceChart
            data={performanceData}
            title="Performance Trends (Last 30 Days)"
          />
        </div>
        <div className="space-y-4">
          <h2 className="text-lg font-medium text-gray-800">Optimization Suggestions</h2>
          <div className="space-y-4 max-h-[400px] overflow-y-auto pr-2">
            {optimizations.map((opt) => (
              <OptimizationCard
                key={opt.id}
                id={opt.id}
                title={opt.title}
                description={opt.description}
                impact={opt.impact}
                campaignName={opt.campaignName}
                onApply={handleApplyOptimization}
                onIgnore={handleIgnoreOptimization}
              />
            ))}
            {optimizations.length === 0 && (
              <div className="text-center py-8 text-gray-500 bg-white rounded-lg border border-gray-200">
                No current optimization suggestions
              </div>
            )}
          </div>
        </div>
      </div>
      
      <div>
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-medium text-gray-800">Recent Campaigns</h2>
          <button className="text-brand-primary hover:text-brand-dark font-medium text-sm">
            View All
          </button>
        </div>
        <DataTable columns={campaignColumns} data={recentCampaigns} />
      </div>
    </div>
  );
};

export default Dashboard;
