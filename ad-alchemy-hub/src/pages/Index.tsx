
import { Navigate } from "react-router-dom";

// This component just redirects to the Dashboard
const Index = () => {
  return <Navigate to="/" replace />;
};

export default Index;
