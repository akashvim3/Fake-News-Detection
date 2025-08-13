import { ErrorBoundary } from "@/components/error-boundary"
import AnalysisDashboard from "@/components/analysis-dashboard"

export default function Home() {
  return (
    <ErrorBoundary>
      <AnalysisDashboard />
    </ErrorBoundary>
  )
}
