import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { content, type } = body

    if (!content || !type) {
      return NextResponse.json({ error: "Content and type are required" }, { status: 400 })
    }

    // Simulate processing delay
    await new Promise((resolve) => setTimeout(resolve, 1000))

    // Generate mock job ID
    const jobId = Math.random().toString(36).substring(2, 15)

    return NextResponse.json({
      job_id: jobId,
      status: "processing",
      message: "Analysis started successfully",
    })
  } catch (error) {
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
