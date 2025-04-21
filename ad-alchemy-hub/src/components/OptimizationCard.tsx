
import React from "react";
import { Check, X } from "lucide-react";

interface OptimizationCardProps {
  id: string;
  title: string;
  description: string;
  impact: string;
  campaignName?: string;
  onApply: (id: string) => void;
  onIgnore: (id: string) => void;
}

const OptimizationCard = ({
  id,
  title,
  description,
  impact,
  campaignName,
  onApply,
  onIgnore,
}: OptimizationCardProps) => {
  return (
    <div className="border rounded-lg p-4 mb-4 bg-white">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="font-medium text-gray-900">{title}</h3>
          {campaignName && (
            <p className="text-sm text-gray-500 mb-1">Campaign: {campaignName}</p>
          )}
          <p className="text-sm text-gray-600 mb-2">{description}</p>
          <div className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-brand-light text-brand-primary">
            Expected Impact: {impact}
          </div>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => onApply(id)}
            className="inline-flex items-center justify-center rounded-md text-sm font-medium bg-brand-primary text-white h-8 w-8 hover:bg-brand-dark transition-colors"
          >
            <Check size={16} />
          </button>
          <button
            onClick={() => onIgnore(id)}
            className="inline-flex items-center justify-center rounded-md text-sm font-medium bg-gray-100 text-gray-500 h-8 w-8 hover:bg-gray-200 transition-colors"
          >
            <X size={16} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default OptimizationCard;
