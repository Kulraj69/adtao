
import React, { useState } from "react";
import DataTable, { Column } from "@/components/DataTable";
import { generateCampaigns, Campaign } from "@/data/mockData";
import { Plus, Filter } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

const Campaigns = () => {
  const [campaigns] = useState<Campaign[]>(generateCampaigns());
  const [filter, setFilter] = useState("all");

  const handleNewCampaign = () => {
    toast.info("New Campaign feature will be available in the next release!");
  };

  const filteredCampaigns = campaigns.filter((campaign) => {
    if (filter === "all") return true;
    return campaign.status === filter;
  });

  const campaignColumns: Column<Campaign>[] = [
    {
      header: "Campaign",
      accessor: "name",
      className: "font-medium text-gray-900",
    },
    {
      header: "Status",
      accessor: (row: Campaign) => (
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
    {
      header: "Platform",
      accessor: (row: Campaign) => (
        <span className="capitalize">{row.platform}</span>
      ),
    },
    {
      header: "Budget",
      accessor: (row: Campaign) => `$${row.budget.toLocaleString()}`,
    },
    {
      header: "Spend",
      accessor: (row: Campaign) => `$${row.spend.toLocaleString()}`,
    },
    { 
      header: "CTR", 
      accessor: (row: Campaign) => `${row.ctr}%` 
    },
    { 
      header: "CPC", 
      accessor: (row: Campaign) => `$${row.cpc.toFixed(2)}` 
    },
    { 
      header: "Conversions", 
      accessor: (row: Campaign) => row.conversions.toString() 
    },
    {
      header: "Actions",
      accessor: () => (
        <div className="flex space-x-2">
          <button className="text-sm text-brand-primary hover:text-brand-dark">
            Edit
          </button>
          <button className="text-sm text-gray-600 hover:text-gray-900">
            View
          </button>
        </div>
      ),
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Campaigns</h1>
          <p className="text-gray-500">Manage your advertising campaigns.</p>
        </div>
        <Button onClick={handleNewCampaign} className="bg-brand-primary hover:bg-brand-dark">
          <Plus size={16} className="mr-2" />
          New Campaign
        </Button>
      </div>

      <div className="bg-white p-4 rounded-lg shadow-sm">
        <div className="flex items-center mb-4 space-x-2">
          <Filter size={16} className="text-gray-500" />
          <span className="text-sm font-medium text-gray-700">Filter:</span>
          <div className="flex space-x-2">
            <button
              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                filter === "all"
                  ? "bg-brand-primary text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
              onClick={() => setFilter("all")}
            >
              All
            </button>
            <button
              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                filter === "active"
                  ? "bg-brand-primary text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
              onClick={() => setFilter("active")}
            >
              Active
            </button>
            <button
              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                filter === "paused"
                  ? "bg-brand-primary text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
              onClick={() => setFilter("paused")}
            >
              Paused
            </button>
            <button
              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                filter === "completed"
                  ? "bg-brand-primary text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
              onClick={() => setFilter("completed")}
            >
              Completed
            </button>
          </div>
        </div>
        <DataTable columns={campaignColumns} data={filteredCampaigns} />
      </div>
    </div>
  );
};

export default Campaigns;
