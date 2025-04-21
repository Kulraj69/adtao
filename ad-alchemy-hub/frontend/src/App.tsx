
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Campaigns from "./pages/Campaigns";
import Creatives from "./pages/Creatives";
import Insights from "./pages/Insights";
import Reports from "./pages/Reports";
import Settings from "./pages/Settings";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <Routes>
            <Route 
              path="/" 
              element={
                <Layout>
                  <Dashboard />
                </Layout>
              } 
            />
            <Route 
              path="/campaigns" 
              element={
                <Layout>
                  <Campaigns />
                </Layout>
              } 
            />
            <Route 
              path="/creatives" 
              element={
                <Layout>
                  <Creatives />
                </Layout>
              } 
            />
            <Route 
              path="/insights" 
              element={
                <Layout>
                  <Insights />
                </Layout>
              } 
            />
            <Route 
              path="/reports" 
              element={
                <Layout>
                  <Reports />
                </Layout>
              } 
            />
            <Route 
              path="/settings" 
              element={
                <Layout>
                  <Settings />
                </Layout>
              } 
            />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </TooltipProvider>
      </BrowserRouter>
    </QueryClientProvider>
  );
};

export default App;
