"use client"

import type React from "react"
import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  FileText,
  ImageIcon,
  Video,
  Mic,
  Upload,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Loader2,
  Shield,
  TrendingUp,
  Users,
  Clock,
} from "lucide-react"

interface AnalysisResult {
  job_id: string
  status: string
  results?: {
    credibility_score: number
    fake_probability: number
    confidence: number
    analysis: {
      text_analysis: {
        sentiment: string
        bias_score: number
        readability: number
      }
      source_analysis: {
        domain_reputation: number
        author_credibility: number
      }
      fact_check: {
        verified_claims: number
        disputed_claims: number
        unverified_claims: number
      }
    }
    recommendations: string[]
  }
}

export default function AnalysisDashboard() {
  const [activeTab, setActiveTab] = useState("text")
  const [content, setContent] = useState("")
  const [file, setFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleAnalyze = async () => {
    if (!content.trim() && !file) {
      setError("Please provide content to analyze")
      return
    }

    setIsAnalyzing(true)
    setError(null)
    setResult(null)

    try {
      // Start analysis
      const analyzeResponse = await fetch("/api/mock/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          content: content || file?.name,
          type: activeTab,
        }),
      })

      if (!analyzeResponse.ok) {
        throw new Error("Failed to start analysis")
      }

      const analyzeData = await analyzeResponse.json()

      // Poll for results
      setTimeout(async () => {
        try {
          const statusResponse = await fetch(`/api/mock/status/${analyzeData.job_id}`)
          if (!statusResponse.ok) {
            throw new Error("Failed to get analysis results")
          }

          const statusData = await statusResponse.json()
          setResult(statusData)
        } catch (err) {
          setError("Failed to get analysis results")
        } finally {
          setIsAnalyzing(false)
        }
      }, 2000)
    } catch (err) {
      setError("Analysis failed. Please try again.")
      setIsAnalyzing(false)
    }
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      setContent("")
    }
  }

  const getCredibilityColor = (score: number) => {
    if (score >= 70) return "text-green-600"
    if (score >= 40) return "text-yellow-600"
    return "text-red-600"
  }

  const getCredibilityIcon = (score: number) => {
    if (score >= 70) return <CheckCircle className="h-5 w-5 text-green-600" />
    if (score >= 40) return <AlertTriangle className="h-5 w-5 text-yellow-600" />
    return <XCircle className="h-5 w-5 text-red-600" />
  }

  return (
    <div className="min-h-screen bg-gray-50 p-4 md:p-6 lg:p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900">FactCheck AI</h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Advanced AI-powered tool for detecting fake news and misinformation across text, images, videos, and audio
            content.
          </p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <Shield className="h-5 w-5 text-blue-600" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Accuracy Rate</p>
                  <p className="text-2xl font-bold text-gray-900">94.2%</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <TrendingUp className="h-5 w-5 text-green-600" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Analyses Today</p>
                  <p className="text-2xl font-bold text-gray-900">1,247</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <Users className="h-5 w-5 text-purple-600" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Active Users</p>
                  <p className="text-2xl font-bold text-gray-900">8,432</p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-2">
                <Clock className="h-5 w-5 text-orange-600" />
                <div>
                  <p className="text-sm font-medium text-gray-600">Avg Response</p>
                  <p className="text-2xl font-bold text-gray-900">2.3s</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Analysis Input */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <FileText className="h-5 w-5" />
                <span>Content Analysis</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Tabs value={activeTab} onValueChange={setActiveTab}>
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="text" className="flex items-center space-x-1">
                    <FileText className="h-4 w-4" />
                    <span className="hidden sm:inline">Text</span>
                  </TabsTrigger>
                  <TabsTrigger value="image" className="flex items-center space-x-1">
                    <ImageIcon className="h-4 w-4" />
                    <span className="hidden sm:inline">Image</span>
                  </TabsTrigger>
                  <TabsTrigger value="video" className="flex items-center space-x-1">
                    <Video className="h-4 w-4" />
                    <span className="hidden sm:inline">Video</span>
                  </TabsTrigger>
                  <TabsTrigger value="audio" className="flex items-center space-x-1">
                    <Mic className="h-4 w-4" />
                    <span className="hidden sm:inline">Audio</span>
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="text" className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Text Content or URL</label>
                    <Textarea
                      placeholder="Paste the text content or URL you want to fact-check..."
                      value={content}
                      onChange={(e) => setContent(e.target.value)}
                      className="min-h-[120px] resize-none"
                    />
                  </div>
                </TabsContent>

                <TabsContent value="image" className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Upload Image</label>
                    <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                      <Upload className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                      <p className="text-sm text-gray-600 mb-2">
                        Upload an image to analyze for manipulation or deepfakes
                      </p>
                      <Input type="file" accept="image/*" onChange={handleFileUpload} className="max-w-xs mx-auto" />
                    </div>
                    {file && <p className="text-sm text-gray-600">Selected: {file.name}</p>}
                  </div>
                </TabsContent>

                <TabsContent value="video" className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Upload Video or URL</label>
                    <Input
                      placeholder="Enter video URL or upload file"
                      value={content}
                      onChange={(e) => setContent(e.target.value)}
                      className="mb-2"
                    />
                    <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                      <Video className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                      <p className="text-sm text-gray-600 mb-2">
                        Upload a video to analyze for deepfakes or manipulation
                      </p>
                      <Input type="file" accept="video/*" onChange={handleFileUpload} className="max-w-xs mx-auto" />
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="audio" className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Upload Audio File</label>
                    <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                      <Mic className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                      <p className="text-sm text-gray-600 mb-2">
                        Upload an audio file to analyze for voice cloning or manipulation
                      </p>
                      <Input type="file" accept="audio/*" onChange={handleFileUpload} className="max-w-xs mx-auto" />
                    </div>
                  </div>
                </TabsContent>
              </Tabs>

              {error && (
                <Alert className="border-red-200 bg-red-50">
                  <AlertTriangle className="h-4 w-4 text-red-600" />
                  <AlertDescription className="text-red-800">{error}</AlertDescription>
                </Alert>
              )}

              <Button onClick={handleAnalyze} disabled={isAnalyzing || (!content.trim() && !file)} className="w-full">
                {isAnalyzing ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Shield className="h-4 w-4 mr-2" />
                    Analyze Content
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Results */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Shield className="h-5 w-5" />
                <span>Analysis Results</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {!result && !isAnalyzing && (
                <div className="text-center py-8">
                  <Shield className="h-12 w-12 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500">Submit content above to see analysis results</p>
                </div>
              )}

              {isAnalyzing && (
                <div className="text-center py-8">
                  <Loader2 className="h-12 w-12 text-blue-600 mx-auto mb-4 animate-spin" />
                  <p className="text-gray-600 mb-2">Analyzing content...</p>
                  <p className="text-sm text-gray-500">This may take a few moments</p>
                </div>
              )}

              {result?.results && (
                <ScrollArea className="h-[400px]">
                  <div className="space-y-6">
                    {/* Overall Score */}
                    <div className="text-center p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center justify-center space-x-2 mb-2">
                        {getCredibilityIcon(result.results.credibility_score)}
                        <span className="text-lg font-semibold">Credibility Score</span>
                      </div>
                      <div className={`text-3xl font-bold ${getCredibilityColor(result.results.credibility_score)}`}>
                        {result.results.credibility_score.toFixed(1)}%
                      </div>
                      <Progress value={result.results.credibility_score} className="mt-2" />
                    </div>

                    {/* Key Metrics */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center p-3 bg-red-50 rounded-lg">
                        <p className="text-sm text-gray-600">Fake Probability</p>
                        <p className="text-xl font-bold text-red-600">
                          {(result.results.fake_probability * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div className="text-center p-3 bg-blue-50 rounded-lg">
                        <p className="text-sm text-gray-600">Confidence</p>
                        <p className="text-xl font-bold text-blue-600">{result.results.confidence.toFixed(1)}%</p>
                      </div>
                    </div>

                    {/* Detailed Analysis */}
                    <div className="space-y-4">
                      <h4 className="font-semibold text-gray-900">Detailed Analysis</h4>

                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-gray-600">Sentiment</span>
                          <Badge
                            variant={
                              result.results.analysis.text_analysis.sentiment === "positive" ? "default" : "secondary"
                            }
                          >
                            {result.results.analysis.text_analysis.sentiment}
                          </Badge>
                        </div>

                        <div className="space-y-1">
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-600">Bias Score</span>
                            <span>{result.results.analysis.text_analysis.bias_score.toFixed(1)}%</span>
                          </div>
                          <Progress value={result.results.analysis.text_analysis.bias_score} />
                        </div>

                        <div className="space-y-1">
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-600">Domain Reputation</span>
                            <span>{result.results.analysis.source_analysis.domain_reputation.toFixed(1)}%</span>
                          </div>
                          <Progress value={result.results.analysis.source_analysis.domain_reputation} />
                        </div>
                      </div>
                    </div>

                    {/* Fact Check Summary */}
                    <div className="space-y-3">
                      <h4 className="font-semibold text-gray-900">Fact Check Summary</h4>
                      <div className="grid grid-cols-3 gap-2 text-center">
                        <div className="p-2 bg-green-50 rounded">
                          <p className="text-lg font-bold text-green-600">
                            {result.results.analysis.fact_check.verified_claims}
                          </p>
                          <p className="text-xs text-gray-600">Verified</p>
                        </div>
                        <div className="p-2 bg-yellow-50 rounded">
                          <p className="text-lg font-bold text-yellow-600">
                            {result.results.analysis.fact_check.disputed_claims}
                          </p>
                          <p className="text-xs text-gray-600">Disputed</p>
                        </div>
                        <div className="p-2 bg-gray-50 rounded">
                          <p className="text-lg font-bold text-gray-600">
                            {result.results.analysis.fact_check.unverified_claims}
                          </p>
                          <p className="text-xs text-gray-600">Unverified</p>
                        </div>
                      </div>
                    </div>

                    {/* Recommendations */}
                    <div className="space-y-3">
                      <h4 className="font-semibold text-gray-900">Recommendations</h4>
                      <ul className="space-y-2">
                        {result.results.recommendations.map((rec, index) => (
                          <li key={index} className="flex items-start space-x-2 text-sm">
                            <CheckCircle className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                            <span className="text-gray-700">{rec}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
