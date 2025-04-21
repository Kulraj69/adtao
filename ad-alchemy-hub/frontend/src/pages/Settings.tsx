
import React from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";

const Settings = () => {
  const handleSaveChanges = () => {
    toast.success("Settings saved successfully!");
  };

  const handleConnectGoogle = () => {
    toast.info("Google Ads connection feature will be available in the next release!");
  };

  const handleConnectFacebook = () => {
    toast.info("Facebook Ads connection feature will be available in the next release!");
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-500">Manage your account and application preferences.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-1">
          <div className="bg-white shadow-sm rounded-lg overflow-hidden">
            <div className="px-4 py-5 sm:px-6 border-b">
              <h3 className="text-lg font-medium text-gray-900">Profile Settings</h3>
            </div>
            <ul className="divide-y divide-gray-200">
              <li>
                <a className="block px-4 py-3 hover:bg-gray-50 cursor-pointer text-brand-primary">
                  Account Information
                </a>
              </li>
              <li>
                <a className="block px-4 py-3 hover:bg-gray-50 cursor-pointer">Password & Security</a>
              </li>
              <li>
                <a className="block px-4 py-3 hover:bg-gray-50 cursor-pointer">Notifications</a>
              </li>
              <li>
                <a className="block px-4 py-3 hover:bg-gray-50 cursor-pointer">Billing</a>
              </li>
              <li>
                <a className="block px-4 py-3 hover:bg-gray-50 cursor-pointer">API Access</a>
              </li>
            </ul>
          </div>
        </div>

        <div className="md:col-span-2">
          <div className="bg-white shadow-sm rounded-lg overflow-hidden">
            <div className="px-6 py-5 border-b">
              <h3 className="text-lg font-medium text-gray-900">Account Information</h3>
            </div>
            <div className="px-6 py-5">
              <div className="grid grid-cols-1 gap-y-6 gap-x-4 sm:grid-cols-6">
                <div className="sm:col-span-3">
                  <label htmlFor="first-name" className="block text-sm font-medium text-gray-700">
                    First name
                  </label>
                  <div className="mt-1">
                    <input
                      type="text"
                      name="first-name"
                      id="first-name"
                      autoComplete="given-name"
                      defaultValue="John"
                      className="shadow-sm focus:ring-brand-primary focus:border-brand-primary block w-full sm:text-sm border-gray-300 rounded-md p-2 border"
                    />
                  </div>
                </div>

                <div className="sm:col-span-3">
                  <label htmlFor="last-name" className="block text-sm font-medium text-gray-700">
                    Last name
                  </label>
                  <div className="mt-1">
                    <input
                      type="text"
                      name="last-name"
                      id="last-name"
                      autoComplete="family-name"
                      defaultValue="Doe"
                      className="shadow-sm focus:ring-brand-primary focus:border-brand-primary block w-full sm:text-sm border-gray-300 rounded-md p-2 border"
                    />
                  </div>
                </div>

                <div className="sm:col-span-4">
                  <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                    Email address
                  </label>
                  <div className="mt-1">
                    <input
                      id="email"
                      name="email"
                      type="email"
                      autoComplete="email"
                      defaultValue="john.doe@example.com"
                      className="shadow-sm focus:ring-brand-primary focus:border-brand-primary block w-full sm:text-sm border-gray-300 rounded-md p-2 border"
                    />
                  </div>
                </div>

                <div className="sm:col-span-3">
                  <label htmlFor="company" className="block text-sm font-medium text-gray-700">
                    Company
                  </label>
                  <div className="mt-1">
                    <input
                      type="text"
                      name="company"
                      id="company"
                      defaultValue="Acme Inc."
                      className="shadow-sm focus:ring-brand-primary focus:border-brand-primary block w-full sm:text-sm border-gray-300 rounded-md p-2 border"
                    />
                  </div>
                </div>

                <div className="sm:col-span-3">
                  <label htmlFor="role" className="block text-sm font-medium text-gray-700">
                    Role
                  </label>
                  <div className="mt-1">
                    <select
                      id="role"
                      name="role"
                      className="shadow-sm focus:ring-brand-primary focus:border-brand-primary block w-full sm:text-sm border-gray-300 rounded-md p-2 border"
                    >
                      <option>Marketing Manager</option>
                      <option>PPC Specialist</option>
                      <option>Agency Owner</option>
                      <option>Other</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>
            <div className="px-6 py-4 bg-gray-50 text-right">
              <Button 
                onClick={handleSaveChanges}
                className="bg-brand-primary hover:bg-brand-dark"
              >
                Save Changes
              </Button>
            </div>
          </div>

          <div className="bg-white shadow-sm rounded-lg overflow-hidden mt-6">
            <div className="px-6 py-5 border-b">
              <h3 className="text-lg font-medium text-gray-900">Connected Ad Accounts</h3>
              <p className="mt-1 text-sm text-gray-500">
                Connect your advertising platform accounts to import campaigns and metrics.
              </p>
            </div>
            <div className="p-6 space-y-4">
              <div className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded bg-red-100 flex items-center justify-center">
                    <span className="text-red-600 font-bold">G</span>
                  </div>
                  <div className="ml-4">
                    <h4 className="text-sm font-medium text-gray-900">Google Ads</h4>
                    <p className="text-xs text-gray-500">Not connected</p>
                  </div>
                </div>
                <Button
                  onClick={handleConnectGoogle}
                  variant="outline"
                  className="border-brand-primary text-brand-primary hover:bg-brand-light"
                >
                  Connect
                </Button>
              </div>
              
              <div className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded bg-blue-100 flex items-center justify-center">
                    <span className="text-blue-600 font-bold">F</span>
                  </div>
                  <div className="ml-4">
                    <h4 className="text-sm font-medium text-gray-900">Facebook Ads</h4>
                    <p className="text-xs text-gray-500">Not connected</p>
                  </div>
                </div>
                <Button
                  onClick={handleConnectFacebook}
                  variant="outline"
                  className="border-brand-primary text-brand-primary hover:bg-brand-light"
                >
                  Connect
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;
