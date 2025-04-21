
import React, { useState } from "react";
import { generateOptimizations, Optimization } from "@/data/mockData";
import { Check, X } from "lucide-react";
import { toast } from "sonner";

const Insights = () => {
  const [optimizations, setOptimizations] = useState<Optimization[]>(
    generateOptimizations()
  );

  const handleApplyOptimization = (id: string) => {
    toast.success("Optimization applied successfully!");
    setOptimizations(optimizations.filter((opt) => opt.id !== id));
  };

  const handleIgnoreOptimization = (id: string) => {
    toast.info("Optimization ignored.");
    setOptimizations(optimizations.filter((opt) => opt.id !== id));
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Optimization Insights</h1>
        <p className="text-gray-500">
          AI-powered recommendations to improve your campaign performance.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {optimizations.map((optimization) => (
          <div
            key={optimization.id}
            className="bg-white rounded-lg shadow-sm overflow-hidden"
          >
            <div className="p-5">
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                {optimization.title}
              </h3>
              {optimization.campaignName && (
                <p className="text-sm text-gray-500 mb-2">
                  Campaign: {optimization.campaignName}
                </p>
              )}
              <p className="text-sm text-gray-600 mb-4">
                {optimization.description}
              </p>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-brand-light text-brand-primary">
                    Expected Impact: {optimization.impact}
                  </span>
                  <span className="ml-3 text-xs text-gray-500">
                    Generated: {optimization.date}
                  </span>
                </div>
                
                <div className="flex space-x-2">
                  <button
                    onClick={() => handleApplyOptimization(optimization.id)}
                    className="inline-flex items-center justify-center rounded-md p-2 text-sm font-medium bg-brand-primary text-white hover:bg-brand-dark"
                  >
                    <Check size={16} className="mr-1" />
                    Apply
                  </button>
                  <button
                    onClick={() => handleIgnoreOptimization(optimization.id)}
                    className="inline-flex items-center justify-center rounded-md p-2 text-sm font-medium bg-gray-100 text-gray-500 hover:bg-gray-200"
                  >
                    <X size={16} className="mr-1" />
                    Ignore
                  </button>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-50 p-4 border-t">
              <div className="flex items-center">
                <div className="w-2 h-2 rounded-full bg-brand-primary mr-2"></div>
                <span className="text-sm font-medium">Impact Forecast</span>
              </div>
              <div className="mt-2 h-12 bg-gray-100 rounded-lg relative overflow-hidden">
                <div 
                  className="absolute top-0 left-0 h-full bg-brand-primary opacity-20"
                  style={{ width: '100%' }}
                ></div>
                <div 
                  className="absolute top-0 left-0 h-full bg-brand-primary"
                  style={{ 
                    width: `${parseInt(optimization.impact) || 10}%`,
                    maxWidth: '100%'
                  }}
                ></div>
              </div>
            </div>
          </div>
        ))}
        
        {optimizations.length === 0 && (
          <div className="col-span-2 text-center py-12 bg-white rounded-lg shadow-sm">
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              No Current Insights
            </h3>
            <p className="text-gray-500">
              Check back later for new AI-powered optimization suggestions.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Insights;
