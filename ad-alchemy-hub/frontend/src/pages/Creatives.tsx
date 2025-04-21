
import React, { useState } from "react";
import { generateCreatives, Creative } from "@/data/mockData";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

const Creatives = () => {
  const [creatives] = useState<Creative[]>(generateCreatives());
  const [activeFilter, setActiveFilter] = useState<string>("all");

  const handleGenerateAd = () => {
    toast.info("AI Ad Generation feature will be available in the next release!");
  };

  const filteredCreatives = creatives.filter(creative => {
    if (activeFilter === "all") return true;
    return creative.type === activeFilter;
  });

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Creative Assets</h1>
          <p className="text-gray-500">
            Manage and optimize your advertising creative assets.
          </p>
        </div>
        <Button 
          onClick={handleGenerateAd} 
          className="bg-brand-primary hover:bg-brand-dark"
        >
          <Plus size={16} className="mr-2" />
          Generate AI Ad
        </Button>
      </div>

      <div className="bg-white p-4 rounded-lg shadow-sm">
        <div className="mb-6">
          <div className="flex space-x-2 mb-4">
            <button
              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                activeFilter === "all"
                  ? "bg-brand-primary text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
              onClick={() => setActiveFilter("all")}
            >
              All
            </button>
            <button
              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                activeFilter === "image"
                  ? "bg-brand-primary text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
              onClick={() => setActiveFilter("image")}
            >
              Images
            </button>
            <button
              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                activeFilter === "video"
                  ? "bg-brand-primary text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
              onClick={() => setActiveFilter("video")}
            >
              Videos
            </button>
            <button
              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                activeFilter === "text"
                  ? "bg-brand-primary text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
              onClick={() => setActiveFilter("text")}
            >
              Text
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredCreatives.map((creative) => (
            <div
              key={creative.id}
              className="border rounded-lg overflow-hidden transition-all hover:shadow-md"
            >
              <div className="h-40 bg-gray-100 relative overflow-hidden">
                <img
                  src={creative.thumbnailUrl}
                  alt={creative.name}
                  className="w-full h-full object-cover"
                />
                <div className="absolute top-2 right-2">
                  <span
                    className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${
                      creative.type === "image"
                        ? "bg-blue-100 text-blue-800"
                        : creative.type === "video"
                        ? "bg-purple-100 text-purple-800"
                        : "bg-gray-100 text-gray-800"
                    }`}
                  >
                    {creative.type}
                  </span>
                </div>
              </div>
              <div className="p-4">
                <h3 className="text-lg font-medium text-gray-900 mb-1">{creative.name}</h3>
                <p className="text-sm text-gray-500 mb-3">
                  Campaign: {creative.campaignName}
                </p>
                <div className="grid grid-cols-3 gap-2 text-sm">
                  <div className="text-center p-2 bg-gray-50 rounded">
                    <div className="font-medium text-gray-900">
                      {creative.impressions.toLocaleString()}
                    </div>
                    <div className="text-xs text-gray-500">Impressions</div>
                  </div>
                  <div className="text-center p-2 bg-gray-50 rounded">
                    <div className="font-medium text-gray-900">
                      {creative.clicks.toLocaleString()}
                    </div>
                    <div className="text-xs text-gray-500">Clicks</div>
                  </div>
                  <div className="text-center p-2 bg-gray-50 rounded">
                    <div className="font-medium text-gray-900">{creative.ctr}%</div>
                    <div className="text-xs text-gray-500">CTR</div>
                  </div>
                </div>
              </div>
              <div className="px-4 py-3 bg-gray-50 border-t flex justify-between items-center">
                <span className="text-xs text-gray-500">
                  Created {creative.createdDate}
                </span>
                <button className="text-sm font-medium text-brand-primary hover:text-brand-dark">
                  Edit
                </button>
              </div>
            </div>
          ))}
        </div>
        
        {filteredCreatives.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-500">No creative assets found matching your filter.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Creatives;
