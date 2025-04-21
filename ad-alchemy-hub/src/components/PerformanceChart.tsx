
import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

interface DataPoint {
  date: string;
  ctr: number;
  cpc: number;
  spend: number;
}

interface PerformanceChartProps {
  data: DataPoint[];
  title: string;
}

const PerformanceChart = ({ data, title }: PerformanceChartProps) => {
  return (
    <div className="bg-white p-4 rounded-lg shadow-sm">
      <h3 className="text-lg font-medium text-gray-800 mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="date" tick={{ fontSize: 12 }} />
          <YAxis yAxisId="left" tick={{ fontSize: 12 }} />
          <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12 }} />
          <Tooltip />
          <Legend />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="ctr"
            name="CTR (%)"
            stroke="#4F46E5"
            activeDot={{ r: 8 }}
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="cpc"
            name="CPC ($)"
            stroke="#10B981"
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="spend"
            name="Spend ($)"
            stroke="#F59E0B"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PerformanceChart;
