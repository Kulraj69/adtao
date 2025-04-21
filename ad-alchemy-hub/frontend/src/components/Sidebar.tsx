
import React from "react";
import { NavLink } from "react-router-dom";
import { 
  LayoutDashboard, 
  BarChart2, 
  Image, 
  Lightbulb, 
  FileText, 
  Settings
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  {
    name: "Dashboard",
    path: "/",
    icon: <LayoutDashboard size={20} />,
  },
  {
    name: "Campaigns",
    path: "/campaigns",
    icon: <BarChart2 size={20} />,
  },
  {
    name: "Creatives",
    path: "/creatives",
    icon: <Image size={20} />,
  },
  {
    name: "Insights",
    path: "/insights",
    icon: <Lightbulb size={20} />,
  },
  {
    name: "Reports",
    path: "/reports",
    icon: <FileText size={20} />,
  },
  {
    name: "Settings",
    path: "/settings",
    icon: <Settings size={20} />,
  },
];

const Sidebar = () => {
  return (
    <div className="h-full bg-white border-r border-gray-200 w-64 flex-shrink-0">
      <div className="p-6">
        <h1 className="text-xl font-bold text-brand-primary">Ad Alchemy Hub</h1>
      </div>
      <nav className="mt-6 px-3">
        {navItems.map((item) => (
          <NavLink
            key={item.name}
            to={item.path}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-all hover:bg-brand-light hover:text-brand-primary my-1",
                isActive
                  ? "bg-brand-light text-brand-primary font-medium"
                  : "text-gray-700"
              )
            }
          >
            {item.icon}
            <span>{item.name}</span>
          </NavLink>
        ))}
      </nav>
    </div>
  );
};

export default Sidebar;
