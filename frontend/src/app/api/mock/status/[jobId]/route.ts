import { type NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest, { params }: { params: { jobId: string } }) {
  try {
    const { jobId } = params

    if (!jobId) {
      return NextResponse.json({ error: "Job ID is required" }, { status: 400 })
    }

    // Simulate processing delay
    await new Promise((resolve) => setTimeout(resolve, 500))

    // Mock analysis results
    const mockResults = {
      job_id: jobId,
      status: "completed",
      results: {
        credibility_score: Math.random() * 100,
        fake_probability: Math.random(),
        confidence: Math.random() * 100,
        analysis: {
          text_analysis: {
            sentiment: Math.random() > 0.5 ? "positive" : "negative",
            bias_score: Math.random() * 100,
            readability: Math.random() * 100,
          },
          source_analysis: {
            domain_reputation: Math.random() * 100,
            author_credibility: Math.random() * 100,
          },
          fact_check: {
            verified_claims: Math.floor(Math.random() * 10),
            disputed_claims: Math.floor(Math.random() * 5),
            unverified_claims: Math.floor(Math.random() * 3),
          },
        },
        recommendations: [
          "Cross-reference with multiple reliable sources",
          "Check the publication date and context",
          "Verify author credentials and expertise",
        ],
      },
    }

    return NextResponse.json(mockResults)
  } catch (error) {
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
