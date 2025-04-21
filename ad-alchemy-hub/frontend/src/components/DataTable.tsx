
import React from "react";
import { cn } from "@/lib/utils";

export interface Column<T> {
  header: string;
  accessor: keyof T | ((row: T) => React.ReactNode);
  className?: string;
}

export interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  className?: string;
}

function DataTable<T extends { id: string | number }>({
  columns,
  data,
  className,
}: DataTableProps<T>) {
  return (
    <div className={cn("bg-white rounded-lg shadow-sm overflow-hidden", className)}>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              {columns.map((column, index) => (
                <th
                  key={index}
                  className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {column.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {data.map((row) => (
              <tr key={row.id} className="hover:bg-gray-50">
                {columns.map((column, index) => (
                  <td
                    key={index}
                    className={cn("px-4 py-3 whitespace-nowrap", column.className)}
                  >
                    {typeof column.accessor === "function"
                      ? column.accessor(row)
                      : (row[column.accessor] as React.ReactNode)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.length === 0 && (
        <div className="text-center py-8 text-gray-500">No data available</div>
      )}
    </div>
  );
}

export default DataTable;
