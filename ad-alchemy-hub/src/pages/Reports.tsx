
import React from "react";
import { FileText, Download, Calendar } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

const Reports = () => {
  const recentReports = [
    {
      id: "rep-1",
      name: "Q2 Performance Summary",
      date: "2023-06-30",
      size: "2.4 MB",
      type: "PDF",
    },
    {
      id: "rep-2",
      name: "Facebook Campaign Analysis",
      date: "2023-06-15",
      size: "1.8 MB",
      type: "HTML",
    },
    {
      id: "rep-3",
      name: "May Monthly Report",
      date: "2023-06-01",
      size: "3.2 MB",
      type: "PDF",
    },
    {
      id: "rep-4",
      name: "Product Launch Results",
      date: "2023-05-20",
      size: "1.5 MB",
      type: "PDF",
    },
  ];

  const handleGenerateReport = () => {
    toast.info("Report generation feature will be available in the next release!");
  };

  const handleDownload = (id: string) => {
    toast.success("Report download initiated!");
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Reports</h1>
        <p className="text-gray-500">
          Generate and download performance reports for your campaigns.
        </p>
      </div>

      <div className="bg-white rounded-lg shadow-sm p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Create New Report</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">Date Range</label>
            <div className="flex items-center">
              <Calendar size={16} className="text-gray-400 mr-2" />
              <select className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm">
                <option>Last 7 days</option>
                <option>Last 30 days</option>
                <option>Last 90 days</option>
                <option>Custom range</option>
              </select>
            </div>
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">Report Type</label>
            <select className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm">
              <option>Performance Summary</option>
              <option>Campaign Analysis</option>
              <option>Creative Performance</option>
              <option>Full Analytics</option>
            </select>
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">Format</label>
            <select className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm">
              <option>PDF</option>
              <option>HTML</option>
              <option>Excel</option>
            </select>
          </div>
        </div>
        
        <Button
          onClick={handleGenerateReport}
          className="bg-brand-primary hover:bg-brand-dark"
        >
          <FileText size={16} className="mr-2" />
          Generate Report
        </Button>
      </div>

      <div className="bg-white rounded-lg shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b">
          <h2 className="text-lg font-medium text-gray-900">Recent Reports</h2>
        </div>
        <div className="divide-y">
          {recentReports.map((report) => (
            <div
              key={report.id}
              className="px-6 py-4 flex items-center justify-between"
            >
              <div className="flex items-center">
                <div
                  className={`w-8 h-8 rounded flex items-center justify-center ${
                    report.type === "PDF"
                      ? "bg-red-100 text-red-600"
                      : "bg-blue-100 text-blue-600"
                  }`}
                >
                  <FileText size={16} />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-900">
                    {report.name}
                  </p>
                  <p className="text-xs text-gray-500">
                    {report.date} • {report.size} • {report.type}
                  </p>
                </div>
              </div>
              <button
                onClick={() => handleDownload(report.id)}
                className="text-sm text-brand-primary hover:text-brand-dark font-medium inline-flex items-center"
              >
                <Download size={16} className="mr-1" />
                Download
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Reports;
