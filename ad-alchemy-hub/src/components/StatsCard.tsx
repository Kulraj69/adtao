
import React from "react";
import { TrendingUp, TrendingDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface StatsCardProps {
  title: string;
  value: string | number;
  trend?: {
    value: number;
    isUpward: boolean;
  };
  className?: string;
}

const StatsCard = ({ title, value, trend, className }: StatsCardProps) => {
  return (
    <div className={cn("stats-card", className)}>
      <div className="stats-card-title">{title}</div>
      <div className="stats-card-value">{value}</div>
      {trend && (
        <div
          className={cn(
            "stats-card-trend",
            trend.isUpward ? "trend-up" : "trend-down"
          )}
        >
          {trend.isUpward ? (
            <TrendingUp size={16} className="mr-1" />
          ) : (
            <TrendingDown size={16} className="mr-1" />
          )}
          <span>{Math.abs(trend.value)}% from last period</span>
        </div>
      )}
    </div>
  );
};

export default StatsCard;
